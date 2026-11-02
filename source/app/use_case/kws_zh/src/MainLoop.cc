/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "BufAttributes.hpp"
#include "Labels.hpp"
#include "UseCaseCommonUtils.hpp"
#include "hal.h"
#include "mlek/fwk/executorch/EtModel.hpp"
#include "mlek/log/log_macros.h"
#include "mlek/use_case/kws_zh/KaldiFbank.hpp"
#include "mlek/use_case/kws_zh/WakeWordDecoder.hpp"
#if defined(KWS_ZH_INPUT_VSI)
#    include "FvpAudioInput.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace arm::app::kws_zh {
extern uint8_t* GetModelPointer();
extern size_t GetModelLen();
#if defined(KWS_ZH_INPUT_VSI)
bool ProcessVsi(fwk::et::EtModel& model, const std::vector<std::string>& labels);
#endif

#if defined(KWS_ZH_INPUT_UART)
bool ProcessFile(fwk::et::EtModel& model, const std::vector<std::string>& labels);
#endif

static uint8_t s_activation[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
#if !defined(KWS_ZH_INPUT_VSI) && !defined(KWS_ZH_INPUT_UART)
static KaldiFbank s_fbank;
#endif

/** Validate the fixed checkpoint contract before accessing tensors. */
static bool CheckModel(const fwk::et::EtModel& model)
{
    const auto input  = model.GetInputTensor(0);
    const auto output = model.GetOutputTensor(0);
    return model.GetNumInputs() == 1 && model.GetNumOutputs() == 1 && input && output &&
           input->Type() == fwk::iface::TensorType::FP32 &&
           output->Type() == fwk::iface::TensorType::FP32 &&
           input->Shape() == std::vector<size_t>{1, KaldiFbank::kFrames, KaldiFbank::kBins} &&
           output->Shape() ==
               std::vector<size_t>{1, KaldiFbank::kFrames, WakeWordDecoder::kKeywords};
}

/** Process static PCM clips with overlapping, independently evaluated windows. */
#if !defined(KWS_ZH_INPUT_VSI) && !defined(KWS_ZH_INPUT_UART)
static bool Process(fwk::et::EtModel& model, const std::vector<std::string>& labels)
{
    Profiler profiler{"kws_zh"};
    WakeWordDecoder decoder{KWS_ZH_SCORE_THRESHOLD,
                            KWS_ZH_MINIMUM_FRAMES,
                            static_cast<uint64_t>(KWS_ZH_COOLDOWN_MS) * 16};
    if (!hal_audio_init() ||
        !hal_audio_configure(HAL_AUDIO_MODE_SINGLE_BURST, HAL_AUDIO_FORMAT_16KHZ_MONO_16BIT)) {
        printf_err("Failed to initialise 16 kHz PCM input\n");
        return false;
    }
    size_t clip = 0;
    while (true) {
        uint32_t count = 0;
        if (!hal_audio_start()) {
            return false;
        }
        const int16_t* audio = hal_audio_get_captured_frame(&count);
        if (!audio || !count) {
            break;
        }
        decoder.Reset();
        info("KWS_ZH clip=%zu samples=%" PRIu32 "\n", clip, count);
        for (size_t start = 0; start < count; start += KWS_ZH_WINDOW_STRIDE) {
            const size_t available = std::min<size_t>(count - start, KaldiFbank::kWindowSamples);
            auto input             = model.GetInputTensor(0);
            profiler.StartProfiling("Fbank");
            const bool preprocessed = s_fbank.Compute(
                audio + start, available, input->GetData<float>(), input->GetNumElements());
            profiler.StopProfiling();
            if (!preprocessed || !RunInference(model, profiler)) {
                printf_err("KWS_ZH processing failed\n");
                return false;
            }
            const auto output   = model.GetOutputTensor(0);
            const float* scores = output->GetData<float>();
            if (!scores) {
                return false;
            }
#    if KWS_ZH_DUMP_OUTPUT
            for (size_t frame = 0; frame < KaldiFbank::kFrames; ++frame) {
                info("KWS_ZH score clip=%zu start=%zu frame=%zu value0=%.9f value1=%.9f\n",
                     clip,
                     start,
                     frame,
                     scores[frame * 2],
                     scores[frame * 2 + 1]);
            }
#    endif
            profiler.StartProfiling("WakeWordDecode");
            const auto events = decoder.Decode(scores, KaldiFbank::kFrames, start, count);
            profiler.StopProfiling();
            for (size_t keyword = 0; keyword < labels.size(); ++keyword) {
                float peak       = 0.f;
                size_t peakFrame = 0;
                for (size_t frame = 0; frame < KaldiFbank::kFrames; ++frame) {
                    const float value = scores[frame * labels.size() + keyword];
                    if (!std::isfinite(value)) {
                        printf_err("Invalid score at frame %zu\n", frame);
                        return false;
                    }
                    if (value > peak) {
                        peak      = value;
                        peakFrame = frame;
                    }
                }
                info("KWS_ZH peak clip=%zu start=%zu label=%s score=%.6f frame=%zu\n",
                     clip,
                     start,
                     labels[keyword].c_str(),
                     peak,
                     peakFrame);
            }
            for (const auto& event : events) {
                info("WAKE label=%s score=%.6f time=%.3fs\n",
                     labels[event.m_keyword].c_str(),
                     event.m_score,
                     static_cast<double>(event.m_sample) / KaldiFbank::kSampleRate);
            }
            if (start + available >= count) {
                break;
            }
        }
        ++clip;
    }
    hal_audio_release();
    profiler.PrintProfilingResult();
    model.LogMemoryUsage();
    return clip > 0;
}
#endif
} // namespace arm::app::kws_zh

void MainLoop()
{
    using namespace arm::app;
    fwk::et::EtModel model;
    fwk::iface::MemoryRegion compute{kws_zh::s_activation, sizeof(kws_zh::s_activation)};
    fwk::iface::MemoryRegion program{kws_zh::GetModelPointer(), kws_zh::GetModelLen()};
    if (!model.Init(compute, program) || !kws_zh::CheckModel(model)) {
        printf_err("Expected WeKWS float32 PTE input [1,198,40], output [1,198,2]\n");
        return;
    }
    std::vector<std::string> labels;
    GetLabelsVector(labels);
    if (labels.size() != kws_zh::WakeWordDecoder::kKeywords ||
        !(KWS_ZH_SCORE_THRESHOLD > 0.f && KWS_ZH_SCORE_THRESHOLD < 1.f)) {
        printf_err("Invalid KWS_ZH labels or score threshold\n");
        return;
    }
    info("WeKWS Chinese KWS: Fbank 40, 25/10 ms, 198 frames, PCM16 scale\n");
#if defined(KWS_ZH_INPUT_VSI)
    const bool success = kws_zh::ProcessVsi(model, labels);
#elif defined(KWS_ZH_INPUT_UART)
    const bool success = kws_zh::ProcessFile(model, labels);
#else
    const bool success = kws_zh::Process(model, labels);
#endif
    info("Main loop terminated %s.\n", success ? "successfully" : "with failure");
#if defined(KWS_ZH_INPUT_VSI)
    if (success) {
        fvp_audio::HoldResult();
    }
#endif
}
