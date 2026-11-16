/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "hal_uart.h"
#include <cstdio>

namespace arm::app::minimind2_small {

constexpr size_t kDemoPromptLimit = kContextLength - 1;

// Each request is one line of decimal token IDs separated by ASCII spaces.
static bool ReadDemoPrompt(int32_t* tokens, size_t& count)
{
    count = 0;
    unsigned value = 0;
    bool haveDigit = false;
    bool valid = true;
    while (true) {
        const auto byte = hal_uart_read_byte();
        if (byte == '?') {
            count = 0;
            value = 0;
            haveDigit = false;
            valid = true;
            std::printf("\nMINIMIND_DEMO HELLO context=%zu max_prompt=%zu\n",
                        kContextLength,
                        kDemoPromptLimit);
            continue;
        }
        if (byte == '\r') {
            continue;
        }
        if (byte >= '0' && byte <= '9') {
            if (value < kVocabularySize) {
                value = value * 10 + byte - '0';
            }
            haveDigit = true;
        } else if (byte == ' ' || byte == '\n') {
            if (haveDigit) {
                if (value >= kVocabularySize || count == kDemoPromptLimit) {
                    valid = false;
                } else {
                    tokens[count++] = static_cast<int32_t>(value);
                }
            }
            value = 0;
            haveDigit = false;
            if (byte == '\n') {
                if (valid && count == 0) {
                    continue;
                }
                return valid;
            }
        } else {
            valid = false;
        }
    }
}

static void RunDemo(fwk::et::EtModel& model)
{
    int32_t prompt[kDemoPromptLimit];
    while (true) {
        std::printf("MINIMIND_DEMO READY context=%zu max_prompt=%zu\n",
                    kContextLength,
                    kDemoPromptLimit);
        size_t promptCount;
        if (!ReadDemoPrompt(prompt, promptCount)) {
            std::printf("MINIMIND_DEMO ERROR invalid_prompt\n");
            continue;
        }

        std::printf("MINIMIND_DEMO ACCEPT prompt_tokens=%zu\n", promptCount);
        ResetCaches();
#ifdef MINIMIND_PROFILE_STAGES
        GetProfiler().Reset();
#endif
        Timing timing{};
        int32_t next = 0;
        bool ok = true;
        for (size_t i = 0; i < promptCount && ok; ++i) {
            ok = Infer(model, prompt[i], i, next, timing, "prompt");
        }
        const Timing prefillTiming = timing;
        const size_t replyLimit = kContextLength - promptCount;
        if (ok) {
            std::printf("MINIMIND_DEMO START prompt_tokens=%zu max_new_tokens=%zu\n",
                        promptCount,
                        replyLimit);
        }
        size_t emitted = 0;
        while (ok && next != 2 && emitted < replyLimit) {
            char tokenBytes[4096];
            size_t tokenByteCount = 0;
            if (!DecodeTokenIds(&next,
                                1,
                                tokenBytes,
                                sizeof(tokenBytes),
                                tokenByteCount)) {
                ok = false;
                break;
            }
            std::printf("MINIMIND_DEMO CHUNK %zu %zu\n", emitted, tokenByteCount);
            hal_uart_write_raw(reinterpret_cast<const uint8_t*>(tokenBytes), tokenByteCount);
            ++emitted;
            if (emitted < replyLimit) {
                ok = Infer(model,
                           next,
                           promptCount + emitted - 1,
                           next,
                           timing,
                           "decode");
            }
        }
        if (!ok) {
            std::printf("MINIMIND_DEMO ERROR inference_or_decode\n");
            continue;
        }

        const uint32_t cpuHz = get_mps4_core_clock();
        const size_t decodeCalls = timing.calls - prefillTiming.calls;
        const double decodeMs = Milliseconds(timing.endToEnd - prefillTiming.endToEnd, cpuHz);
        // Infer resets the PMU on every call. Use only the accumulated durations.
        std::printf("\nMINIMIND_DEMO TIMING prefill_calls=%zu prefill_infer_ms=%.3f "
                    "decode_calls=%zu decode_infer_ms=%.3f decode_ms_per_call=%.3f\n",
                    prefillTiming.calls,
                    Milliseconds(prefillTiming.endToEnd, cpuHz),
                    decodeCalls,
                    decodeMs,
                    decodeCalls ? decodeMs / decodeCalls : 0.0);
        std::printf("MINIMIND_BREAKDOWN npu_hz=%u cpu_hz=%" PRIu32
                    " prompt_npu_cycles=%" PRIu64 " decode_npu_cycles=%" PRIu64
                    " prompt_e2e_cycles=%" PRIu64 " decode_e2e_cycles=%" PRIu64 "\n",
                    static_cast<unsigned>(MINIMIND_NPU_CLOCK_HZ),
                    cpuHz, prefillTiming.npu, timing.npu - prefillTiming.npu,
                    prefillTiming.endToEnd, timing.endToEnd - prefillTiming.endToEnd);
#ifdef MINIMIND_PROFILE_STAGES
        GetProfiler().Print(true);
#endif
        std::printf("MINIMIND_DEMO DONE tokens=%zu reason=%s\n",
                    emitted,
                    next == 2 ? "eos" : "context_limit");
    }
}

} // namespace arm::app::minimind2_small
