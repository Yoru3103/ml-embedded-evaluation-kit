/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "MiniMindReference.hpp"
#include "MiniMindRuntime.hpp"
#ifdef MINIMIND_PROFILE_STAGES
#include "MiniMindProfiler.hpp"
#include "MiniMindQuantProfile.inc"
#endif

#include "hal.h"
#include "mlek/log/log_macros.h"

#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>

namespace arm::app::minimind2_small {

static bool SetInt32Input(fwk::et::EtModel& model, size_t index, int32_t value)
{
    const auto tensor = model.GetInputTensor(index);
    if (!tensor || tensor->Type() != fwk::iface::TensorType::INT32 ||
        tensor->GetNumElements() != 1 || tensor->GetData<int32_t>() == nullptr) {
        printf_err("MiniMind2 input %zu must be a scalar INT32 tensor\n", index);
        return false;
    }
    tensor->GetData<int32_t>()[0] = value;
    return true;
}

static bool SetInt64Input(fwk::et::EtModel& model, size_t index, int64_t value)
{
    const auto tensor = model.GetInputTensor(index);
    if (!tensor || tensor->Bytes() != sizeof(int64_t) || tensor->GetNumElements() != 1 ||
        tensor->GetData<int64_t>() == nullptr) {
        printf_err("MiniMind2 input %zu must be a scalar INT64 tensor\n", index);
        return false;
    }
    tensor->GetData<int64_t>()[0] = value;
    return true;
}

static bool GetTopToken(const fwk::et::EtModel& model, int32_t& token)
{
    const auto tensor = model.GetOutputTensor(0);
    if (!tensor || tensor->Type() != fwk::iface::TensorType::FP32 ||
        tensor->GetNumElements() != kVocabularySize || tensor->GetData<float>() == nullptr) {
        printf_err("MiniMind2 output must be a %zu-element FP32 logits tensor\n",
                   kVocabularySize);
        return false;
    }

    const float* logits = tensor->GetData<float>();
    float maximum       = -std::numeric_limits<float>::infinity();
    size_t topIndex     = 0;
    for (size_t i = 0; i < tensor->GetNumElements(); ++i) {
        if (!std::isfinite(logits[i])) {
            printf_err("MiniMind2 produced a non-finite logit at %zu\n", i);
            return false;
        }
        if (logits[i] > maximum) {
            maximum  = logits[i];
            topIndex = i;
        }
    }
    token = static_cast<int32_t>(topIndex);
    return true;
}

static void PrintTop5(const fwk::et::EtModel& model, size_t position)
{
    const auto tensor = model.GetOutputTensor(0);
    if (!tensor || tensor->Type() != fwk::iface::TensorType::FP32 ||
        tensor->GetNumElements() != kVocabularySize || tensor->GetData<float>() == nullptr) {
        return;
    }
    const float* logits = tensor->GetData<float>();
    float values[5] = {-std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity()};
    int32_t ids[5] = {-1, -1, -1, -1, -1};
    for (size_t index = 0; index < kVocabularySize; ++index) {
        for (int slot = 0; slot < 5; ++slot) {
            if (logits[index] > values[slot]) {
                for (int move = 4; move > slot; --move) {
                    values[move] = values[move - 1];
                    ids[move]   = ids[move - 1];
                }
                values[slot] = logits[index];
                ids[slot]    = static_cast<int32_t>(index);
                break;
            }
        }
    }
    info("MINIMIND_TOP5 position=%zu ids=%" PRId32 ",%" PRId32 ",%" PRId32 ",%" PRId32
         ",%" PRId32 " values=%.6f,%.6f,%.6f,%.6f,%.6f\n",
         position,
         ids[0],
         ids[1],
         ids[2],
         ids[3],
         ids[4],
         values[0],
         values[1],
         values[2],
         values[3],
         values[4]);
}

static uint64_t Counter(const pmu_counters& counters, const char* name)
{
    for (uint32_t i = 0; i < counters.num_counters; ++i) {
        if (std::strcmp(counters.counters[i].name, name) == 0) {
            return counters.counters[i].value;
        }
    }
    return 0;
}

uint64_t CpuCycles()
{
    pmu_counters counters{};
    hal_pmu_get_cpu_counters(&counters);
    return Counter(counters, "CPU TOTAL");
}

double Milliseconds(uint64_t cycles, uint32_t frequency)
{
    return static_cast<double>(cycles) * 1000.0 / frequency;
}

void PrintTiming(const char* phase, size_t position, const Timing& timing)
{
    const uint32_t cpuHz = get_mps4_core_clock();
    info("MINIMIND_TIME phase=%s position=%zu calls=%zu "
         "npu_total_cycles=%" PRIu64 " npu_total_ms=%.3f "
         "invoke_cycles=%" PRIu64 " invoke_ms=%.3f "
         "e2e_cycles=%" PRIu64 " e2e_ms=%.3f\n",
         phase,
         position,
         timing.calls,
         timing.npu,
         Milliseconds(timing.npu, MINIMIND_NPU_CLOCK_HZ),
         timing.invoke,
         Milliseconds(timing.invoke, cpuHz),
         timing.endToEnd,
         Milliseconds(timing.endToEnd, cpuHz));
}

bool Infer(fwk::et::EtModel& model,
           int32_t token,
           size_t position,
           int32_t& nextToken,
           Timing& total,
           const char* phase)
{
    hal_pmu_reset();
    const uint64_t begin = CpuCycles();
    if (!SetInt64Input(model, 0, token) ||
        !SetInt32Input(model, 1, static_cast<int32_t>(position))) {
        hal_pmu_final();
        return false;
    }
    const uint64_t invokeBegin = CpuCycles();
#ifdef MINIMIND_PROFILE_STAGES
    auto& profiler = GetProfiler();
    profiler.Begin(phase, invokeBegin);
#endif
    const bool invoked         = model.RunInference();
    const uint64_t invokeEnd   = CpuCycles();
#ifdef MINIMIND_PROFILE_STAGES
    profiler.Finish(invokeEnd);
#endif
    const bool valid           = invoked && GetTopToken(model, nextToken);
    if (valid && position <= 2) {
        PrintTop5(model, position);
    }
    const uint64_t end         = CpuCycles();
    pmu_counters counters{};
    hal_pmu_get_counters(&counters);
    hal_pmu_final();
    const Timing timing{Counter(counters, "NPU TOTAL"), invokeEnd - invokeBegin,
                        end - begin, 1};
    PrintTiming(phase, position, timing);

    // Functional FVP --fast may not simulate the NPU or CPU PMU counters.
    // Only FPGA requires a nonzero hardware NPU count; CPU profiling is optional.
#ifdef MINIMIND_FVP
    const bool npuTimingValid = true;
#else
    const bool npuTimingValid = timing.npu != 0;
#endif
    if (!valid || !npuTimingValid) {
        printf_err("MiniMind2 inference/timing failed at position %zu\n", position);
        return false;
    }

    total.npu += timing.npu;
    total.invoke += timing.invoke;
    total.endToEnd += timing.endToEnd;
    ++total.calls;
    return true;
}
} // namespace arm::app::minimind2_small

#ifdef MINIMIND_PROFILE_STAGES
extern "C" decltype(ethosu_invoke_v3) __real_ethosu_invoke_v3;

extern "C" int __wrap_ethosu_invoke_v3(struct ethosu_driver* driver,
                                      const void* commands,
                                      const int commandSize,
                                      uint64_t* const bases,
                                      const size_t* sizes,
                                      const int count,
                                      void* argument)
{
    using namespace arm::app::minimind2_small;
    const auto begin = CpuCycles();
    const int result =
        __real_ethosu_invoke_v3(driver, commands, commandSize, bases, sizes, count, argument);
    GetProfiler().RecordDriver(begin, CpuCycles());
    return result;
}
#endif
