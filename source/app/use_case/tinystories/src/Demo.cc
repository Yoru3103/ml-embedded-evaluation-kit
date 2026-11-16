/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#ifdef TINYSTORIES_DEMO
#    include "TinyStoriesCacheContract.hpp"
#    include "TinyStoriesRuntime.hpp"
#    include "TokenDecoder.hpp"
#    include "TokenEncoder.hpp"
#    include "hal.h"
#    include "hal_uart.h"
#    include "mlek/log/log_macros.h"
#    include <algorithm>
#    include <cinttypes>
#    include <cmath>
#    include <cstdlib>
#    include <cstring>

#    ifndef TINYSTORIES_CONTEXT
#        define TINYSTORIES_CONTEXT 64
#    endif

namespace arm::app::tinystories {
namespace {
    constexpr size_t kContext           = TINYSTORIES_CONTEXT;
    constexpr uint32_t kDefaultSeedBase = 0x5EED0000U;
    struct Workspace {
        char prompt[kMaxPromptBytes + 1];
        char text[4096];
        int32_t tokens[kContext];
        InferenceRecord records[kContext];
        DecodeContext decode;
    };
    static Workspace s_workspace
        __attribute__((section(".bss.NoInit.activation_buf_sram"), aligned(16)));
    uint32_t s_seedBase     = kDefaultSeedBase;
    uint32_t s_requestIndex = 0;

    bool ParseUint32(const char* text, uint32_t& value)
    {
        char* end                  = nullptr;
        const unsigned long parsed = std::strtoul(text, &end, 0);
        if (end == text || *end != '\0' || parsed > 0xFFFFFFFFUL) {
            return false;
        }
        value = static_cast<uint32_t>(parsed);
        return true;
    }

    bool ParseFloat(const char* text, float& value)
    {
        char* end          = nullptr;
        const float parsed = std::strtof(text, &end);
        if (end == text || *end != '\0' || !std::isfinite(parsed)) {
            return false;
        }
        value = parsed;
        return true;
    }

    void PrintDecode()
    {
        const SamplingConfig& config = s_workspace.decode.sampling;
        if (config.greedy) {
            info("TINYSTORIES_DEMO decode=greedy\n");
        } else {
            info("TINYSTORIES_DEMO decode=sample temperature=%.2f top_p=%.2f top_k=%zu\n",
                 config.temperature,
                 config.topP,
                 config.topK);
        }
    }

    bool ApplyCommand(const char* line)
    {
        SamplingConfig& config = s_workspace.decode.sampling;
        bool valid             = true;
        if (std::strcmp(line, "!greedy") == 0) {
            config.greedy = true;
        } else if (std::strcmp(line, "!sample") == 0) {
            config.greedy = false;
        } else if (std::strncmp(line, "!seed ", 6) == 0) {
            valid = ParseUint32(line + 6, s_seedBase);
        } else if (std::strncmp(line, "!temp ", 6) == 0) {
            valid = ParseFloat(line + 6, config.temperature) && config.temperature > 0.0F;
        } else if (std::strncmp(line, "!topp ", 6) == 0) {
            valid = ParseFloat(line + 6, config.topP) && config.topP > 0.0F && config.topP <= 1.0F;
        } else if (std::strncmp(line, "!topk ", 6) == 0) {
            uint32_t topK = 0;
            valid = ParseUint32(line + 6, topK) && topK > 0 && topK <= kMaxSamplingCandidates;
            if (valid) {
                config.topK = topK;
            }
        } else {
            info("TINYSTORIES_DEMO commands: !greedy !sample !seed N !temp F !topp F !topk N\n");
            return true;
        }
        if (!valid) {
            info("TINYSTORIES_DEMO rejected command: %s\n", line);
            return true;
        }
        PrintDecode();
        return true;
    }

    void Write(const char* data, size_t bytes)
    { hal_uart_write_raw(reinterpret_cast<const uint8_t*>(data), bytes); }

    bool ReadPrompt(char* output, size_t capacity)
    {
        static bool skipLf = false;
        size_t size        = 0;
        bool overflow      = false;
        for (;;) {
            const uint8_t byte = hal_uart_read_byte();
            if (skipLf && byte == '\n') {
                skipLf = false;
                continue;
            }
            skipLf = false;
            if (byte == '\r' || byte == '\n') {
                skipLf       = byte == '\r';
                output[size] = '\0';
                Write("\r\n", 2);
                return !overflow && size != 0;
            }
            if (byte == 8 || byte == 127) {
                if (size && !overflow) {
                    --size;
                    while (size && (static_cast<uint8_t>(output[size]) & 0xC0) == 0x80) {
                        --size;
                    }
                    Write("\b \b", 3);
                }
            } else if (byte >= 0x20 && byte != 127 && !overflow) {
                if (size + 1 == capacity) {
                    overflow = true; // Drain the complete line; never run a truncated prompt.
                } else {
                    output[size++] = static_cast<char>(byte);
                    Write(reinterpret_cast<const char*>(&byte), 1);
                }
            }
        }
    }

    bool ResetCache(fwk::et::EtModel& model)
    {
        const auto& allocator = model.GetBackendData().m_plannedMemAllocPtr;
        if (!allocator) {
            return false;
        }
        for (const auto& region : kCacheRegions) {
            auto address = allocator->get_offset_address(0, region.offset, region.bytes);
            if (!address.ok()) {
                return false;
            }
            std::memset(address.get(), 0, region.bytes);
        }
        return true;
    }

    bool Stream(size_t tokenCount, bool finish, size_t& printed, uint64_t& cycles)
    {
        size_t bytes         = 0;
        const uint64_t begin = CpuCycles();
        const bool valid     = DecodeTokens(
            s_workspace.tokens, tokenCount, s_workspace.text, sizeof(s_workspace.text), bytes);
        cycles += CpuCycles() - begin;
        if (!valid || bytes < printed) {
            return false;
        }
        size_t boundary = printed;
        for (size_t index = printed; index < bytes; ++index) {
            const char value = s_workspace.text[index];
            if (value == '.' || value == '!' || value == '?' || value == '\n') {
                boundary = index + 1;
            }
        }
        if (finish) {
            boundary = bytes;
        }
        Write(s_workspace.text + printed, boundary - printed);
        printed = boundary;
        return true;
    }

    bool Generate(fwk::et::EtModel& model, size_t promptCount, uint64_t requestBegin, uint32_t seed)
    {
        const bool reset = ResetCache(model);
        if (!reset) {
            printf_err("TINYSTORIES_DEMO reset failed\n");
            return false;
        }
        const size_t budget = kContext - promptCount;
        info("TINYSTORIES_DEMO BUSY prompt_tokens=%zu max_new_tokens=%zu cache_reset=1\n",
             promptCount,
             budget);
        Write("\r\nStory:\r\n", 10);
        Timing promptTiming{}, decodeTiming{};
        size_t calls = 0;
        int32_t next = -1;
        bool success = true;
        for (size_t position = 0; position < promptCount; ++position) {
            if (!Infer(model,
                       s_workspace.tokens[position],
                       position,
                       next,
                       promptTiming,
                       "prompt",
                       &s_workspace.records[calls],
                       &s_workspace.decode)) {
                success = false;
                break;
            }
            ++calls;
        }
        size_t generated = 0, printed = 0;
        uint64_t textCycles = 0;
        const char* reason  = "context_limit";
        while (success && generated < budget) {
            s_workspace.tokens[promptCount + generated++] = next;
            success = Stream(promptCount + generated, false, printed, textCycles);
            if (next == 2) {
                reason = "eos";
                break;
            }
            if (!success || generated == budget) {
                break;
            }
            const size_t position = promptCount + generated - 1;
            success               = Infer(model,
                                          next,
                                          position,
                                          next,
                                          decodeTiming,
                                          "decode",
                                          &s_workspace.records[calls],
                                          &s_workspace.decode);
            if (success) {
                ++calls;
            }
        }
        success = Stream(promptCount + generated, true, printed, textCycles) && success;
        const uint64_t totalCycles = CpuCycles() - requestBegin;
        Write("\r\n\r\n", 4);
        bool stagesValid = true;
        for (size_t position = 0; position < calls; ++position) {
            const auto& record = s_workspace.records[position];
            s_profiler->SetSnapshot(record.stages);
            stagesValid =
                s_profiler->Print(position, record.timing.invoke, get_mps4_core_clock(), false) &&
                stagesValid;
        }
        info("TINYSTORIES_DEMO DONE success=%d total_ms=%.3f prompt_tokens=%zu generated=%zu "
             "reason=%s seed=%" PRIu32 "\n",
             success && stagesValid,
             Milliseconds(totalCycles, get_mps4_core_clock()),
             promptCount,
             generated,
             success && stagesValid ? reason : "error",
             seed);
        return success && stagesValid;
    }
} // namespace

void RunDemo(fwk::et::EtModel& model)
{
#    ifdef TINYSTORIES_FVP
    constexpr const char* platform = "FVP";
    info("TINYSTORIES_DEMO functional_FVP=1 timings_are_not_FPGA_performance "
         "npu_pmu_zero_allowed=1\n");
#    else
    constexpr const char* platform = "FPGA";
#    endif
    SamplingConfig& config = s_workspace.decode.sampling;
    config.greedy          = false;
    config.temperature     = 0.8F;
    config.topP            = 0.9F;
    config.topK            = kMaxSamplingCandidates;
    info("TINYSTORIES_DEMO mode=%s context=%zu max_new_tokens=remaining_context "
         "cpu_hz=%" PRIu32 " npu_hz=%u\n",
         platform,
         kContext,
         get_mps4_core_clock(),
         static_cast<unsigned>(TINYSTORIES_NPU_CLOCK_HZ));
    PrintDecode();
    info("Commands: !greedy !sample !seed N !temp F !topp F !topk N\n");
    info("Enter an English story opening (UTF-8, <=256 bytes, <=%zu tokens including BOS). "
         "115200 8N1; local echo OFF; CR, LF or CRLF.\n",
         kContext - 1);
    info("Sentences stream during generation; only total time is printed afterward. "
         "Total includes encoding, KV reset, inference, token sampling, decoding and "
         "story UART output.\n");
    for (;;) {
        info("TINYSTORIES_DEMO READY\n");
        Write("Prompt> ", 8);
        if (!ReadPrompt(s_workspace.prompt, sizeof(s_workspace.prompt))) {
            info("TINYSTORIES_DEMO rejected: empty or oversized prompt\n");
            continue;
        }
        if (s_workspace.prompt[0] == '!') {
            ApplyCommand(s_workspace.prompt);
            continue;
        }
        size_t promptCount = 0;
        hal_pmu_reset();
        const uint64_t begin = CpuCycles();
        const bool encoded =
            EncodePrompt(s_workspace.prompt, s_workspace.tokens, kContext - 1, promptCount);
        hal_pmu_final();
        if (!encoded) {
            info("TINYSTORIES_DEMO rejected: invalid UTF-8 or prompt exceeds token limit\n");
            continue;
        }
        const uint32_t seed    = s_seedBase + s_requestIndex++;
        s_workspace.decode.rng = XorShift32(seed);
        if (!Generate(model, promptCount, begin, seed)) {
            printf_err(
                "TINYSTORIES_DEMO stopped after runtime/contract failure; reboot required\n");
            return;
        }
    }
}
} // namespace arm::app::tinystories
#endif
