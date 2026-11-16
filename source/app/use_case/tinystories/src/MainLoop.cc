/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "BufAttributes.hpp"
#include "TinyStoriesRuntime.hpp"
#ifdef TINYSTORIES_DEMO
#    include "TinyStoriesCacheContract.hpp"
#endif
#include "TinyStoriesReference.hpp"
#include "TokenDecoder.hpp"
#include "hal.h"
#include "mlek/fwk/executorch/EtModel.hpp"
#include "mlek/log/log_macros.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>

#ifndef TINYSTORIES_CONTEXT
#    define TINYSTORIES_CONTEXT 64
#endif

namespace arm::app::tinystories {
extern uint8_t* GetModelPointer();
extern size_t GetModelLen();

static uint8_t s_methodPool[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;

} // namespace arm::app::tinystories

void MainLoop()
{
    using namespace arm::app;
    setvbuf(stdout, nullptr, _IONBF, 0);
    fwk::et::EtModel model;
    fwk::et::EtBackendData backend;
    backend.m_logTensorDetails = false;
    static tinystories::TinyStoriesProfiler profiler(tinystories::CpuCycles);
    tinystories::s_profiler = &profiler;
    backend.m_eventTracer   = &profiler;
    if (!profiler.CheckEventHooks()) {
        printf_err("TinyStories runtime profiling hooks are inactive\n");
        return;
    }
    fwk::iface::MemoryRegion methodPool{tinystories::s_methodPool,
                                        sizeof(tinystories::s_methodPool)};
    fwk::iface::MemoryRegion program{tinystories::GetModelPointer(), tinystories::GetModelLen()};
#ifdef TINYSTORIES_DEMO
    static executorch::runtime::Span<uint8_t> plannedSpans[] = {
        {tinystories::s_methodPool, tinystories::kPlannedCacheArenaBytes}};
    backend.m_plannedMemAllocPtr = std::make_shared<executorch::runtime::HierarchicalAllocator>(
        executorch::runtime::Span{plannedSpans, size_t{1}});
#endif
    if (!model.Init(methodPool, program, &backend) || !model.ContainsEthosUOperator() ||
        model.GetNumInputs() != 2 || model.GetNumOutputs() != 1) {
        printf_err("TinyStories failed model, NPU delegate, or I/O initialization\n");
        return;
    }
#ifdef TINYSTORIES_CACHE_CLAMP_SPECIALIZED
    info("TinyStories: specialized in-place INT16 KV clamp enabled\n");
#endif
#ifdef TINYSTORIES_DEMO
    tinystories::RunDemo(model);
    return;
#else
    info("TinyStories PTE bytes=%zu; greedy regression, context=%d, max_new_tokens=16\n",
         program.size,
         TINYSTORIES_CONTEXT);
    info("TINYSTORIES_CLOCK cpu_hz=%" PRIu32 " npu_hz=%u "
         "npu_clock_source=configured_FPGA_assumption\n",
         get_mps4_core_clock(),
         static_cast<unsigned>(TINYSTORIES_NPU_CLOCK_HZ));
    info("TinyStories timing: NPU TOTAL=hardware cycles including NPU stalls; "
         "invoke=RunInference; e2e=input+invoke+argmax; "
         "model init, PMU setup, UART and separately measured text decoding excluded\n");

    info("TinyStories stages: CPU kernels + Delegate.Total + Runtime.Other = invoke; "
         "Delegate.Total contains InputCopy + NPU.Driver + OutputCopy + Delegate.Other; "
         "Driver uses CPU elapsed cycles, NPU TOTAL uses hardware cycles\n");
    fflush(stdout);
    tinystories::Timing promptTiming{}, decodeTiming{};
    size_t promptMatches = 0;
    int32_t nextToken    = -1;
    for (size_t position = 0; position < std::size(tinystories::kPromptTokens); ++position) {
        const int32_t token = tinystories::kPromptTokens[position];
        info("TINYSTORIES_LOOP position=%zu token=%" PRId32 "\n", position, token);
        if (!tinystories::Infer(model, token, position, nextToken, promptTiming, "prompt")) {
            return;
        }
        const int32_t expected = tinystories::kPromptExpected[position];
        promptMatches += nextToken == expected;
        info("TINYSTORIES position=%zu input=%" PRId32 " top1=%" PRId32 " expected=%" PRId32
             " match=%d\n",
             position,
             token,
             nextToken,
             expected,
             nextToken == expected);
    }
    info("TINYSTORIES_SMOKE matches=%zu total=%zu success=%d last_expected=%" PRId32
         " last_actual=%" PRId32 " reference=host_quantized_PT2\n",
         promptMatches,
         std::size(tinystories::kPromptTokens),
         promptMatches == std::size(tinystories::kPromptTokens),
         tinystories::kPromptExpected[std::size(tinystories::kPromptTokens) - 1],
         nextToken);

    constexpr size_t kGeneratedTokens = TINYSTORIES_CONTEXT - std::size(tinystories::kPromptTokens);
    constexpr size_t kReferenceTokens = std::size(tinystories::kGeneratedExpected);
    int32_t generated[kGeneratedTokens]{};
    size_t generatedCount    = 0;
    size_t generationMatches = 0;
    for (size_t index = 0; index < std::size(generated); ++index) {
        generated[generatedCount++] = nextToken;
        const int32_t expected      = index < kReferenceTokens ? tinystories::kGeneratedExpected[index] : -1;
        const bool match            = expected >= 0 && nextToken == expected;
        generationMatches += match;
        info("TINYSTORIES_GENERATED index=%zu token=%" PRId32 " expected=%" PRId32 " match=%d\n",
             index,
             nextToken,
             expected,
             match);
        if (nextToken == 2 || index + 1 == std::size(generated)) {
            break;
        }
        const size_t position = std::size(tinystories::kPromptTokens) + index;
        if (position >= TINYSTORIES_CONTEXT ||
            !tinystories::Infer(model, nextToken, position, nextToken, decodeTiming, "decode")) {
            return;
        }
    }
    const size_t referenceCount = kGeneratedTokens < kReferenceTokens ? kGeneratedTokens : kReferenceTokens;
    info("TINYSTORIES_GENERATION matches=%zu count=%zu reference_count=%zu success=%d\n",
         generationMatches,
         generatedCount,
         referenceCount,
         generatedCount >= referenceCount && generationMatches == referenceCount);
    tinystories::PrintTiming("prompt_sum", 0, promptTiming);
    tinystories::PrintTiming("decode_sum", 0, decodeTiming);
    int32_t textTokens[std::size(tinystories::kPromptTokens) + std::size(generated)]{};
    std::memcpy(textTokens, tinystories::kPromptTokens, sizeof(tinystories::kPromptTokens));
    std::memcpy(textTokens + std::size(tinystories::kPromptTokens),
                generated,
                generatedCount * sizeof(int32_t));
    static char textBuffer[4096]
        __attribute__((section(".bss.NoInit.activation_buf_sram"), aligned(16)));
    size_t textBytes         = 0;
    const uint64_t textBegin = tinystories::CpuCycles();
    const bool textOk =
        tinystories::DecodeTokens(textTokens,
                                  std::size(tinystories::kPromptTokens) + generatedCount,
                                  textBuffer,
                                  sizeof(textBuffer),
                                  textBytes);
    const uint64_t textCycles = tinystories::CpuCycles() - textBegin;
    info("TINYSTORIES_TEXT_TIME cycles=%" PRIu64 " ms=%.3f bytes=%zu success=%d uart_excluded=1\n",
         textCycles,
         tinystories::Milliseconds(textCycles, get_mps4_core_clock()),
         textBytes,
         textOk);
    if (textOk) {
        info("TINYSTORIES_TEXT %s\n", textBuffer);
    } else {
        printf_err("TinyStories text decoding failed (invalid token/table or buffer capacity)\n");
    }
    model.LogMemoryUsage();
#endif
}
