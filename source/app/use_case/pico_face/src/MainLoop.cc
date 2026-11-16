/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "BufAttributes.hpp"
#include "PicoFaceParams.hpp"
#include "hal.h"
#include "mlek/fwk/executorch/EtProgram.hpp"
#include "mlek/log/log_macros.h"
#include "mlek/use_case/pico_face/PicoFaceProcessing.hpp"
#include "mlek/use_case/pico_face/PicoFaceSampler.hpp"
#include "mlek/use_case/pico_face/PicoFaceSerial.hpp"
#if PICO_FACE_UART_IMAGE
#    include "hal_uart.h"
#endif
#include <array>
#include <cstdio>
#include <cstring>
#include <initializer_list>

namespace arm::app::pico_face {
/** Read the existing SysTick clock; NPU-only resets leave this running. */
static uint64_t CpuCycles(const pmu_counters& counters)
{
    for (uint32_t i = 0; i < counters.num_counters; ++i) {
        if (std::strcmp(counters.counters[i].name, "CPU TOTAL") == 0) {
            return counters.counters[i].value;
        }
    }
    return 0;
}

static uint64_t ReadCycles()
{
    pmu_counters counters{};
    hal_pmu_get_cpu_counters(&counters);
    return CpuCycles(counters);
}

static uint64_t Microseconds(uint64_t cycles)
{
    const uint32_t clock = get_mps4_core_clock();
    return (cycles / clock) * 1000000U + (cycles % clock) * 1000000U / clock;
}

extern uint8_t* GetModelPointer();
extern size_t GetModelLen();

static uint8_t s_methodPool[ACTIVATION_BUF_SZ] MAKE_ATTRIBUTE(section("activation_buf_dram"));
static PicoFaceSampler s_sampler MAKE_ATTRIBUTE(section("activation_buf_dram"));
/** Kept as global symbols so a debugger can retrieve the final RGB on hardware. */
uint8_t g_image[kImageElements] MAKE_ATTRIBUTE(section("activation_buf_dram"));

class EtPicoFaceModel final : public PicoFaceModel {
public:
    EtPicoFaceModel() :
        m_program(GetModelPointer(),
                  GetModelLen(),
                  s_methodPool,
                  sizeof(s_methodPool),
                  reinterpret_cast<uint8_t*>(PICO_FACE_TEMP_BASE),
                  PICO_FACE_TEMP_SIZE)
    {}

    bool Init()
    {
        static_assert(PF_LATENT_CH == 8 && PF_LATENT_HW == 16 && PF_COND_DIM == 128);
        static_assert(PF_IMG_CH == 3 && PF_IMG_HW == 128 && PF_K_MAX == 8 && PF_N_COND == 5);
        return m_program.LoadMethod("dit_step") && m_program.LoadMethod("decode") &&
               CheckShape("dit_step", true, 0, {1, 8, 16, 16}) &&
               CheckShape("dit_step", true, 1, {1, 128}) &&
               CheckShape("dit_step", false, 0, {1, 8, 16, 16}) &&
               CheckShape("decode", true, 0, {1, 8, 16, 16}) &&
               CheckShape("decode", false, 0, {1, 3, 128, 128});
    }

    bool DitStep(float* latent, const float* condition, float* velocity) override
    {
        using namespace executorch::aten;
        TensorImpl z(ScalarType::Float, 4, m_latentSizes.data(), latent, m_latentOrder.data());
        TensorImpl c(ScalarType::Float,
                     2,
                     m_condSizes.data(),
                     const_cast<float*>(condition),
                     m_condOrder.data());
        pmu_counters start{};
        hal_pmu_reset_npu();
        hal_pmu_get_counters(&start);
        auto result = m_program.Execute("dit_step", {Tensor(&z), Tensor(&c)});
        RecordMethod(start, "dit_step", m_ditCycles, m_ditCalls);
        if (!result.ok() || !result->isTensor()) {
            return false;
        }
        std::memcpy(
            velocity, result->toTensor().const_data_ptr<float>(), kLatentElements * sizeof(float));
        return true;
    }

    const float* Decode(float* latent) override
    {
        using namespace executorch::aten;
        TensorImpl z(ScalarType::Float, 4, m_latentSizes.data(), latent, m_latentOrder.data());
        pmu_counters start{};
        hal_pmu_reset_npu();
        hal_pmu_get_counters(&start);
        auto result = m_program.Execute("decode", {Tensor(&z)});
        RecordMethod(start, "decode", m_decodeCycles, m_decodeCalls);
        return result.ok() && result->isTensor() ? result->toTensor().const_data_ptr<float>()
                                                 : nullptr;
    }

    void PrintStatistics(const GenerationConfig& config)
    {
        m_program.LogMemoryUsage();
        info("Euler steps: %u\n", config.steps);
        info("DiT calls: %u\n", m_ditCalls);
        info("Decode calls: %u\n", m_decodeCalls);
        unsigned ditCall  = 0;
        const bool guided = config.classId != 4 && config.guidance > 0;
        for (size_t i = 0; i < m_recordCount; ++i) {
            const auto& record = m_records[i];
            if (std::strcmp(record.name, "dit_step") == 0) {
                const unsigned step = guided ? ditCall / 2 + 1 : ditCall + 1;
                const char* branch  = config.classId == 4 || (guided && ditCall % 2)
                                          ? "unconditional"
                                          : "conditional";
                info("Profile for dit_step, step %u, %s:\n", step, branch);
                ++ditCall;
            } else {
                info("Profile for decode:\n");
            }
            for (uint32_t j = 0; j < record.counters.num_counters; ++j) {
                const auto& counter = record.counters.counters[j];
                if (std::strcmp(counter.name, "DURATION") != 0) {
                    info("%s: %" PRIu64 " %s\n", counter.name, counter.value, counter.unit);
                }
            }
            info("DURATION: %.3f milliseconds\n",
                 static_cast<double>(Microseconds(CpuCycles(record.counters))) / 1000.0);
        }
    }

private:
    void
    RecordMethod(const pmu_counters& start, const char* name, uint64_t& cycles, unsigned& calls)
    {
        pmu_counters end{};
        hal_pmu_get_counters(&end);
        hal_pmu_final();
        cycles += CpuCycles(end) - CpuCycles(start);
        ++calls;
        if (m_recordCount < m_records.size() && start.num_counters == end.num_counters) {
            auto& record    = m_records[m_recordCount++];
            record.name     = name;
            record.counters = end;
            for (uint32_t i = 0; i < end.num_counters; ++i) {
                record.counters.counters[i].value -= start.counters[i].value;
            }
        }
    }

    bool CheckShape(const char* method,
                    bool input,
                    size_t index,
                    std::initializer_list<int32_t> expected)
    {
        auto meta = m_program.Metadata(method);
        if (!meta.ok()) {
            return false;
        }
        auto tensor = input ? meta->input_tensor_meta(index) : meta->output_tensor_meta(index);
        if (!tensor.ok() || tensor->scalar_type() != executorch::aten::ScalarType::Float ||
            tensor->sizes().size() != expected.size()) {
            return false;
        }
        size_t i = 0;
        for (auto dimension : expected) {
            if (tensor->sizes()[i++] != dimension) {
                return false;
            }
        }
        return true;
    }

    fwk::et::EtProgram m_program;
    struct MethodRecord {
        const char* name{nullptr};
        pmu_counters counters{};
    };
    /* Eight guided steps, the final decode and up to eight preview decodes. */
    std::array<MethodRecord, PICO_FACE_PREVIEW ? 25 : 17> m_records{};
    size_t m_recordCount{0};
    uint64_t m_ditCycles{0};
    uint64_t m_decodeCycles{0};
    unsigned m_ditCalls{0};
    unsigned m_decodeCalls{0};
    std::array<executorch::aten::SizesType, 4> m_latentSizes{1, 8, 16, 16};
    std::array<executorch::aten::DimOrderType, 4> m_latentOrder{0, 1, 2, 3};
    std::array<executorch::aten::SizesType, 2> m_condSizes{1, 128};
    std::array<executorch::aten::DimOrderType, 2> m_condOrder{0, 1};
};
#if PICO_FACE_SAVE_IMAGE
static bool SaveRgb(const char* path)
{
    FILE* image = std::fopen(path, "wb");
    if (!image) {
        printf_err("Cannot open %s\n", path);
        return false;
    }
    const size_t written = std::fwrite(g_image, 1, sizeof(g_image), image);
    const int closed     = std::fclose(image);
    if (written != sizeof(g_image) || closed != 0) {
        printf_err("Cannot save %s\n", path);
        return false;
    }
    return true;
}
#endif

#if PICO_FACE_PREVIEW
static bool ReadPreviewConfig(GenerationConfig& config)
{
    FILE* input = std::fopen("pico_face_preview.bin", "rb");
    if (!input) {
        printf_err("Cannot open pico_face_preview.bin\n");
        return false;
    }
    uint8_t bytes[5];
    const size_t count = std::fread(bytes, 1, sizeof(bytes), input);
    const bool exactSize =
        count == sizeof(bytes) && std::fgetc(input) == EOF && !std::ferror(input);
    const int closed = std::fclose(input);
    if (!exactSize || closed != 0 || bytes[4] > 4) {
        printf_err("Invalid pico_face_preview.bin\n");
        return false;
    }
    config.seed = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
                  (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
    config.classId = bytes[4];
    return true;
}

static void PublishFrame(unsigned step)
{
    info("PICO_FACE_FRAME step=%u crc=%08x\n",
         step,
         static_cast<unsigned>(ImageCrc32(g_image, sizeof(g_image))));
    std::fflush(stdout);
}

static bool PreviewStep(PicoFaceModel& model, const float* latent, unsigned step)
{
    static float s_previewLatent[kLatentElements] MAKE_ATTRIBUTE(section("activation_buf_dram"));
    std::memcpy(s_previewLatent, latent, sizeof(s_previewLatent));
    const float* image = model.Decode(s_previewLatent);
    if (!image) {
        return false;
    }
    ConvertToRgb(image, g_image, kImageWidth * kImageWidth);
    char path[24];
    std::snprintf(path, sizeof(path), "pico_face_%u.rgb", step);
    if (!SaveRgb(path)) {
        return false;
    }
    PublishFrame(step);
    return true;
}
#endif
} // namespace arm::app::pico_face

void MainLoop()
{
    using namespace arm::app::pico_face;
    const uint64_t firstStart = ReadCycles();
    EtPicoFaceModel model;
    if (!model.Init()) {
        printf_err("pico_face initialization failed\n");
        return;
    }
    const uint64_t initEnd = ReadCycles();
    GenerationConfig config{PICO_FACE_SEED,
                            PICO_FACE_STEPS,
                            PICO_FACE_CLASS_ID,
                            static_cast<float>(PICO_FACE_GUIDANCE)};
#if PICO_FACE_PREVIEW
    if (!ReadPreviewConfig(config)) {
        return;
    }
#endif
    info("Generating seed=%u steps=%u class=%u guidance=%.1f\n",
         config.seed,
         config.steps,
         config.classId,
         static_cast<double>(config.guidance));
    const uint64_t generationStart = ReadCycles();
#if PICO_FACE_PREVIEW
    constexpr auto kObserve = PreviewStep;
#else
    constexpr auto kObserve = nullptr;
#endif
    if (!s_sampler.Generate(model, config, &pf_cond[0][0][0], pf_schedule, g_image, kObserve)) {
        printf_err("pico_face generation failed\n");
        return;
    }
    const uint64_t generationEnd = ReadCycles();
    const uint64_t generationUs  = Microseconds(generationEnd - generationStart);
    info("Timing core clock: %" PRIu32 " Hz\n", get_mps4_core_clock());
    info("Profile for model initialization:\n");
    info("CPU TOTAL: %" PRIu64 " cycles\n", initEnd - firstStart);
    info("DURATION: %.3f milliseconds\n",
         static_cast<double>(Microseconds(initEnd - firstStart)) / 1000.0);
    info("Profile for image generation (%s):\n",
         PICO_FACE_PREVIEW ? "including intermediate previews" : "noise to RGB");
    info("CPU TOTAL: %" PRIu64 " cycles\n", generationEnd - generationStart);
    info("DURATION: %.3f milliseconds\n", static_cast<double>(generationUs) / 1000.0);
    info("Profile for first image (including model initialization):\n");
    info("CPU TOTAL: %" PRIu64 " cycles\n", generationEnd - firstStart);
    info("DURATION: %.3f milliseconds\n",
         static_cast<double>(Microseconds(generationEnd - firstStart)) / 1000.0);
    const uint32_t crc = ImageCrc32(g_image, sizeof(g_image));
    info("RGB888: 128x128, %zu bytes, CRC32=%08x, address=%p\n",
         sizeof(g_image),
         static_cast<unsigned>(crc),
         g_image);
#if PICO_FACE_SAVE_IMAGE
    if (!SaveRgb("pico_face.rgb")) {
        return;
    }
#endif
#if PICO_FACE_PREVIEW
    PublishFrame(config.steps);
#endif
    model.PrintStatistics(config);
    info("PICO_FACE_PASS\n");
#if PICO_FACE_UART_IMAGE
    info("SERIAL_IMAGE_READY: send I to retrieve RGB, Q to exit\n");
    std::fflush(stdout);
    for (;;) {
        const uint8_t command = hal_uart_read_byte();
        if (command == 'Q') {
            break;
        }
        if (command != 'I') {
            continue;
        }
        const auto header = BuildImageHeader(config, crc, generationUs);
        std::fflush(stdout);
        const uint64_t transferStart = ReadCycles();
        hal_uart_write_raw(header.data(), header.size());
        hal_uart_write_raw(g_image, sizeof(g_image));
        const uint64_t transferEnd = ReadCycles();
        info("\nSERIAL_IMAGE_SENT: enqueue_us=%" PRIu64 "\n",
             Microseconds(transferEnd - transferStart));
        std::fflush(stdout);
    }
#endif
}
