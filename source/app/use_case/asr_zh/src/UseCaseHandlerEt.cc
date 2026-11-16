/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#if ASR_ZH_BILINGUAL
#    include "TokenBytesBilingual.hpp"
#else
#    include "TokenBytes.hpp"
#endif
#include "UseCaseHandler.hpp"
#include "Validation.hpp"
#include "hal.h"
#include "mlek/log/log_macros.h"
#include "mlek/use_case/asr_zh/KaldiFbank.hpp"
#include "mlek/use_case/asr_zh/StreamingState.hpp"
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#if ASR_ZH_INT8_STATE || ASR_ZH_INT16_STATE
#    include "StateQuantization.hpp"
#endif
#if ASR_ZH_PROFILE
#    include "AsrZhOpProfiler.hpp"
#endif

namespace arm::app::asr_zh {
namespace {
    KaldiFbank s_fbank;

    bool DumpOutputs(fwk::et::EtModel& model, size_t clip, size_t chunk, bool inputs = false)
    {
#if ASR_ZH_DUMP_TENSORS
        if (chunk < 3) {
            char name[64];
            std::snprintf(name,
                          sizeof(name),
                          "asr_zh_%03zu_%03zu_%s.bin",
                          clip,
                          chunk,
                          inputs ? "inputs" : "outputs");
            auto* file = std::fopen(name, "wb");
            if (!file) {
                return false;
            }
            bool success       = true;
            const size_t count = inputs ? model.GetNumInputs() : model.GetNumOutputs();
            for (size_t slot = 0; slot < count; ++slot) {
                const auto tensor =
                    inputs ? model.GetInputTensor(slot) : model.GetOutputTensor(slot);
                success = success && std::fwrite(tensor->GetData(), 1, tensor->Bytes(), file) ==
                                         tensor->Bytes();
            }
            const int status = std::fclose(file);
            return success && status == 0;
        }
#else
        (void)model;
        (void)clip;
        (void)chunk;
        (void)inputs;
#endif
        return true;
    }

    uint64_t Cycles()
    {
#if ASR_ZH_PROFILE
        return ReadCycles();
#else
        return 0;
#endif
    }

    bool BindStates(fwk::et::EtModel& model, StreamingState& states)
    {
        using fwk::iface::TensorType;
        if (model.GetNumInputs() != 75 || model.GetNumOutputs() != 75) {
            return false;
        }
        const auto in  = model.GetInputTensor(0);
        const auto out = model.GetOutputTensor(0);
        if (!in || !out || in->Type() != TensorType::FP32 || out->Type() != TensorType::FP32 ||
            in->Shape() != std::vector<size_t>{1, 77, 80} ||
            out->Shape() != std::vector<size_t>{1, 16, ASR_ZH_VOCAB_SIZE}) {
            return false;
        }
        for (size_t slot = 1; slot <= StreamingState::kCount; ++slot) {
            const auto input  = model.GetInputTensor(slot);
            const auto output = model.GetOutputTensor(slot);
            if (!input || !output) {
                return false;
            }
            const auto storage = slot == 74           ? StateStorage::FLOAT32
                                 : ASR_ZH_INT16_STATE ? StateStorage::INT16
                                 : ASR_ZH_INT8_STATE  ? StateStorage::INT8
                                                      : StateStorage::FLOAT32;
            const auto type    = slot == 74                       ? TensorType::INT32
                                 : storage == StateStorage::INT16 ? TensorType::INT16
                                 : storage == StateStorage::INT8  ? TensorType::INT8
                                                                  : TensorType::FP32;
            int32_t zeroPoint  = 0;
#if ASR_ZH_INT8_STATE
            if (storage == StateStorage::INT8) {
                zeroPoint = state_quant::kZeroPoints[slot - 1];
                if (input->GetQuantParams().offset != zeroPoint ||
                    output->GetQuantParams().offset != zeroPoint ||
                    input->GetQuantParams().scale != state_quant::kScales[slot - 1] ||
                    output->GetQuantParams().scale != state_quant::kScales[slot - 1]) {
                    printf_err("ASR_ZH state quantization metadata mismatch slot=%zu\n", slot);
                    return false;
                }
            }
#endif
#if ASR_ZH_INT16_STATE
            if (storage == StateStorage::INT16 &&
                ((input->GetQuantParams().scale > 0 &&
                  input->GetQuantParams().scale != state_quant::kScales[slot - 1]) ||
                 (output->GetQuantParams().scale > 0 &&
                  output->GetQuantParams().scale != state_quant::kScales[slot - 1]))) {
                printf_err("ASR_ZH state quantization metadata mismatch slot=%zu\n", slot);
                return false;
            }
#endif
            if (!input || !output || input->Type() != type || output->Type() != type ||
                input->Shape() != StateShape(slot, ASR_ZH_LEFT_CONTEXT) ||
                output->Shape() != StateShape(slot, ASR_ZH_LEFT_CONTEXT) ||
                !states.Bind(slot,
                             input->GetData(),
                             input->Bytes(),
                             output->GetData(),
                             output->Bytes(),
                             storage,
                             zeroPoint,
                             ASR_ZH_LEFT_CONTEXT)) {
                printf_err("ASR_ZH invalid state slot=%zu\n", slot);
                return false;
            }
        }
        return true;
    }

#if ASR_ZH_VALIDATE && !ASR_ZH_BILINGUAL
    [[maybe_unused]] void PrintValidation(fwk::et::EtModel& model,
                                          size_t chunk,
                                          const float* features,
                                          const CtcDecoder& decoder)
    {
        info("ASR_ZH_PATH chunk=%zu", chunk);
        const auto& path = decoder.Path();
        for (size_t i = path.size() - 16; i < path.size(); ++i) {
            printf(" %u", path[i]);
        }
        printf("\n");
        if (chunk >= sizeof(assets::kExpectedPath) / sizeof(assets::kExpectedPath[0])) {
            return;
        }
        size_t matches = 0;
        for (size_t i = 0; i < 16; ++i) {
            matches += path[path.size() - 16 + i] == assets::kExpectedPath[chunk][i];
        }
        float featureError = 0.f;
        if (chunk < 3) {
            for (size_t i = 0; i < 77 * 80; ++i) {
                featureError = std::max(featureError,
                                        std::abs(features[i] - assets::kFixedFeatures[chunk][i]));
            }
        }
        float maxStateError = 0.f;
        for (size_t slot = 1; slot < 74; ++slot) {
            const auto tensor      = model.GetOutputTensor(slot);
            const size_t count     = tensor->GetNumElements();
            const size_t indices[] = {0, count / 2, count - 1};
            for (size_t i = 0; i < 3; ++i) {
#    if ASR_ZH_INT8_STATE
                const float actual = (static_cast<int32_t>(tensor->GetData<int8_t>()[indices[i]]) -
                                      state_quant::kZeroPoints[slot - 1]) *
                                     state_quant::kScales[slot - 1];
#    else
                const float actual = tensor->GetData<float>()[indices[i]];
#    endif
                maxStateError = std::max(
                    maxStateError, std::abs(actual - assets::kExpectedStates[chunk][slot - 1][i]));
            }
        }
        info("ASR_ZH_VALIDATION chunk=%zu argmax=%zu/16 feature_max_abs=%.8f "
             "sampled_state_max_abs=%.8f\n",
             chunk,
             matches,
             featureError,
             maxStateError);
    }
#endif
} // namespace

bool ProcessAudio(fwk::et::EtModel& model)
{
    StreamingState states;
    if (!BindStates(model, states)) {
        printf_err("ASR_ZH requires the exact 75/75 streaming Zipformer contract\n");
        return false;
    }
#if (defined(ASR_ZH_INPUT_VSI) || defined(ASR_ZH_INPUT_UART))
    if (!OpenDemoInput()) {
        return false;
    }
#else
    if (!hal_audio_init() ||
        !hal_audio_configure(HAL_AUDIO_MODE_SINGLE_BURST, HAL_AUDIO_FORMAT_16KHZ_MONO_16BIT)) {
        return false;
    }
#endif
    size_t clips = 0;
    bool success = true;
    for (;;) {
#if !(defined(ASR_ZH_INPUT_VSI) || defined(ASR_ZH_INPUT_UART))
        if (!hal_audio_start()) {
            break;
        }
#endif
        uint32_t count = 0;
#if (defined(ASR_ZH_INPUT_VSI) || defined(ASR_ZH_INPUT_UART))
        const auto* pcm = ReadDemoClip(count);
#else
        const auto* pcm = hal_audio_get_captured_frame(&count);
#endif
        if (!pcm || !count) {
            break;
        }
        const uint64_t setupBegin = Cycles();
        CtcDecoder decoder{assets::kTokens, ASR_ZH_VOCAB_SIZE, ASR_ZH_BILINGUAL};
        if (!states.Reset() || !s_fbank.Begin(pcm, count)) {
            success = false;
#if defined(ASR_ZH_INPUT_VSI) || defined(ASR_ZH_INPUT_UART)
            CompleteDemoClip(false);
#endif
            break;
        }
        const uint64_t setup = Cycles() - setupBegin;
        auto input           = model.GetInputTensor(0);
        float* features      = input->GetData<float>();
        size_t chunk         = 0;
        uint64_t total       = setup;
        [[maybe_unused]] const size_t chunks =
            (s_fbank.Frames() + KaldiFbank::kAdvance - 1) / KaldiFbank::kAdvance;
        [[maybe_unused]] uint64_t totalPre = 0, totalInvoke = 0, totalPost = 0, totalCopy = 0;
#if ASR_ZH_PROFILE
        [[maybe_unused]] uint64_t totalQuantize = 0, totalNpu = 0, totalDequantize = 0;
#    if ASR_ZH_TIMING_DETAILED
        info("ASR_ZH timing: CPU elapsed cycles; NPU.Driver includes submission/wait; "
             "UART/display/validation/dump excluded\n");
#    endif
#endif
        while (chunk * KaldiFbank::kAdvance < s_fbank.Frames()) {
#if ASR_ZH_FIXED_FEATURES
            if (chunk >= 3) {
                break;
            }
#endif
            const uint64_t begin = Cycles();
#if ASR_ZH_FIXED_FEATURES
            std::copy_n(assets::kFixedFeatures[chunk], 77 * 80, features);
#else
            if (!s_fbank.Next(features, input->GetNumElements())) {
                success = false;
                break;
            }
#endif
            const uint64_t pre = Cycles() - begin;
            if (!DumpOutputs(model, clips, chunk, true)) {
                printf_err("ASR_ZH input dump failed\n");
                success = false;
                break;
            }
#if ASR_ZH_PROFILE
            g_profiler->Reset();
#endif
            const uint64_t invokeBegin = Cycles();
            const bool invoked         = model.RunInference();
            const uint64_t invoke      = Cycles() - invokeBegin;
#if ASR_ZH_PROFILE
            g_profiler->Stop();
#endif
            if (!invoked) {
                printf_err("ASR_ZH_FAILED clip=%zu chunk=%zu stage=Invoke "
                           "completed_chunks=%zu\n",
                           clips,
                           chunk + 1,
                           chunk);
#if ASR_ZH_TIMING_DETAILED
                using Operation           = AsrZhOpProfiler::Operation;
                const uint64_t quantize   = g_profiler->Get(Operation::QUANTIZE).cycles;
                const uint64_t npu        = g_profiler->Get(Operation::ETHOS_U).cycles;
                const uint64_t dequantize = g_profiler->Get(Operation::DEQUANTIZE).cycles;
                const uint64_t measured   = quantize + npu + dequantize;
                AsrZhOpProfiler::PrintStage("Failed.Pre", pre);
                AsrZhOpProfiler::PrintStage("Failed.Q", quantize);
                AsrZhOpProfiler::PrintStage("Failed.NPU", npu);
                AsrZhOpProfiler::PrintStage("Failed.DQ", dequantize);
                AsrZhOpProfiler::PrintStage("Failed.Other",
                                            invoke >= measured ? invoke - measured : 0);
                AsrZhOpProfiler::PrintStage("Failed.Invoke", invoke);
#endif
                success = false;
                break;
            }
            const uint64_t postBegin = Cycles();
            if (!decoder.Append(
                    model.GetOutputTensor(0)->GetData<float>(), 16, ASR_ZH_VOCAB_SIZE)) {
                success = false;
                break;
            }
            const uint64_t post   = Cycles() - postBegin;
            const auto* processed = model.GetOutputTensor(74)->GetData<int32_t>();
            if (*processed != static_cast<int32_t>((chunk + 1) * 32)) {
                printf_err("ASR_ZH state counter mismatch\n");
                success = false;
                break;
            }
#if ASR_ZH_VALIDATE && !ASR_ZH_BILINGUAL
            PrintValidation(model, chunk, features, decoder);
#endif
            if (!DumpOutputs(model, clips, chunk)) {
                printf_err("ASR_ZH output dump failed\n");
                success = false;
                break;
            }
#if ASR_ZH_DUMP_TENSORS
            info("ASR_ZH_REGRESSION clip=%zu chunk=%zu path=", clips, chunk);
            const auto& framePath = decoder.Path();
            for (size_t i = framePath.size() - 16; i < framePath.size(); ++i) {
                printf("%u ", framePath[i]);
            }
            printf("\n");
#endif
            const uint64_t copyBegin = Cycles();
            if (!states.Advance()) {
                success = false;
                break;
            }
            const uint64_t copy = Cycles() - copyBegin;
            total += pre + invoke + post + copy;
            totalPre += pre;
            totalInvoke += invoke;
            totalPost += post;
            totalCopy += copy;
#if ASR_ZH_PROFILE
            using Operation = AsrZhOpProfiler::Operation;
            totalQuantize += g_profiler->Get(Operation::QUANTIZE).cycles;
            totalNpu += g_profiler->Get(Operation::ETHOS_U).cycles;
            totalDequantize += g_profiler->Get(Operation::DEQUANTIZE).cycles;
#    if ASR_ZH_TIMING_DETAILED
            const bool validTiming = g_profiler->Print(chunk, chunks, pre, invoke, post, copy);
#    else
            const bool validTiming = g_profiler->Validate(invoke);
#    endif
            if (!validTiming) {
                success = false;
                break;
            }
#endif
            ++chunk;
        }
        const uint64_t textBegin  = Cycles();
        const std::string text    = decoder.Text();
        const uint64_t textCycles = Cycles() - textBegin;
        total += textCycles;
#if ASR_ZH_TIMING_SUMMARY
        const uint64_t frequency = get_mps4_core_clock();
        if (frequency) {
            const auto toUs = [frequency](uint64_t cycles) {
                return cycles / frequency * 1000000ULL +
                       cycles % frequency * 1000000ULL / frequency;
            };
            printf("ASR_ZH_TIMING end_to_end_us=%" PRIu64 " npu_us=%" PRIu64 "\n",
                   toUs(total),
                   toUs(totalNpu));
        } else {
            success = false;
        }
#endif
#if ASR_ZH_TIMING_DETAILED
        info(
            "ASR_ZH timing: summary chunks=%zu cpu_hz=%" PRIu32 "\n", chunk, get_mps4_core_clock());
        AsrZhOpProfiler::PrintStage("Clip.Setup", setup);
        AsrZhOpProfiler::PrintStage("Preprocess", totalPre);
        AsrZhOpProfiler::PrintStage("Quantize", totalQuantize);
        AsrZhOpProfiler::PrintStage("NPU.Driver", totalNpu);
        AsrZhOpProfiler::PrintStage("Dequantize", totalDequantize);
        const uint64_t measured = totalQuantize + totalNpu + totalDequantize;
        if (measured > totalInvoke) {
            success = false;
        }
        AsrZhOpProfiler::PrintStage("Invoke.Other",
                                    totalInvoke >= measured ? totalInvoke - measured : 0);
        AsrZhOpProfiler::PrintStage("Postprocess", totalPost + textCycles);
        AsrZhOpProfiler::PrintStage("State.Copy", totalCopy);
        AsrZhOpProfiler::PrintStage("Decode.Final", textCycles);
        AsrZhOpProfiler::PrintStage("Invoke.Total", totalInvoke);
        AsrZhOpProfiler::PrintStage("Pipeline.Total", total);
#endif
        info("ASR_ZH_TEXT clip=%zu chunks=%zu text=%s\n", clips, chunk, text.c_str());
#if !defined(ASR_ZH_INPUT_UART) || ASR_ZH_TIMING_DETAILED
        info("ASR_ZH_TOTAL samples=%u cycles=%" PRIu64 " text_cycles=%" PRIu64
             " computed_feature_frames=%zu\n",
             count,
             total,
             textCycles,
             s_fbank.ComputedFrames());
#endif
#if ASR_ZH_VALIDATE && !ASR_ZH_FIXED_FEATURES && !ASR_ZH_BILINGUAL
        info("ASR_ZH_TEXT_REFERENCE match=%d\n", text == assets::kExpectedText);
        success = success && text == assets::kExpectedText;
#endif
#if defined(ASR_ZH_INPUT_VSI) || defined(ASR_ZH_INPUT_UART)
        CompleteDemoClip(success);
#endif
        ++clips;
        if (!success) {
            break;
        }
    }
#if (defined(ASR_ZH_INPUT_VSI) || defined(ASR_ZH_INPUT_UART))
    success = CloseDemoInput() && success;
    ShowDemoStatus(success ? "Complete - Chinese in terminal" : "Input / inference failed");
#else
    hal_audio_release();
#endif
    return success && clips > 0;
}
} // namespace arm::app::asr_zh
