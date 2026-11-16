/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "BufAttributes.hpp"
#include "MiniMindReference.hpp"
#include "MiniMindRuntime.hpp"
#ifdef MINIMIND_PROFILE_STAGES
#include "MiniMindProfiler.hpp"
#endif
#include "MiniMindTokenDecoder.hpp"
#include "hal.h"
#include "mlek/fwk/executorch/EtModel.hpp"
#include "mlek/log/log_macros.h"

#include <cinttypes>
#include <cstring>
#include <iterator>

#if MINIMIND_CONTEXT_LENGTH == 128 && !defined(MINIMIND_CACHE_OFFSETS)
#error "The 128-token PTE requires its cache offsets to be supplied at build time"
#endif

namespace arm::app::minimind2_small {
extern uint8_t* GetModelPointer();
extern size_t GetModelLen();

static uint8_t s_methodPool[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
constexpr size_t kCacheOffsets[] = {
#ifdef MINIMIND_CACHE_OFFSETS
    MINIMIND_CACHE_OFFSETS,
#else
    917504, 950272, 983040, 1015808, 1048576, 1081344, 1114112, 1146880,
    1179648, 1212416, 1245184, 1277952, 1310720, 1343488, 1376256, 1409024,
#endif
};
#ifdef MINIMIND_INT16_KV
constexpr size_t kCacheBytes = kContextLength * 256;
#else
constexpr size_t kCacheBytes = kContextLength * 512;
#endif
static_assert(std::size(kCacheOffsets) == 16);
static_assert(kCacheOffsets[15] + kCacheBytes <= sizeof(s_methodPool));

static void ResetCaches()
{
    for (size_t offset : kCacheOffsets) {
        std::memset(s_methodPool + offset, 0, kCacheBytes);
    }
}

#ifndef MINIMIND_INT16_KV
static void DumpCaches(const char* phase)
{
    for (size_t index = 0; index < std::size(kCacheOffsets); ++index) {
        const float* cache =
            reinterpret_cast<const float*>(s_methodPool + kCacheOffsets[index]);
        info("MINIMIND_CACHE phase=%s index=%zu offset=%zu "
             "v0=%.6f %.6f %.6f %.6f "
             "v1h0=%.6f %.6f %.6f %.6f "
             "v1h1=%.6f %.6f %.6f %.6f\n",
             phase,
             index,
             kCacheOffsets[index],
             cache[0],
             cache[1],
             cache[2],
             cache[3],
             cache[64],
             cache[65],
             cache[66],
             cache[67],
             cache[4160],
             cache[4161],
             cache[4162],
             cache[4163]);
    }
}
#endif
} // namespace arm::app::minimind2_small

#ifdef MINIMIND_DEMO
#include "MiniMindDemo.inc"
#endif

void MainLoop()
{
    using namespace arm::app;
    fwk::et::EtModel model;
    fwk::et::EtBackendData backend;
    backend.m_logTensorDetails = false;
    fwk::iface::MemoryRegion methodPool{minimind2_small::s_methodPool,
                                        sizeof(minimind2_small::s_methodPool)};
    fwk::iface::MemoryRegion program{minimind2_small::GetModelPointer(),
                                    minimind2_small::GetModelLen()};
    if (!model.Init(methodPool, program, &backend) || !model.ContainsEthosUOperator() ||
        model.GetNumInputs() != 2 || model.GetNumOutputs() != 1) {
        printf_err("MiniMind2 failed model, NPU delegate, or I/O initialization\n");
        return;
    }
    // The exported PTE retains the calibration cache in its mutable data.
    // A new request must start with empty KV state. Diagnostic builds that
    // compare against a host run which also keeps the calibration cache set
    // MINIMIND_NO_CACHE_RESET instead.
#ifndef MINIMIND_NO_CACHE_RESET
    minimind2_small::ResetCaches();
#endif

#ifdef MINIMIND_DEMO
    info("MiniMind2 PTE bytes=%zu; greedy generation, context=%zu, dynamic reply budget\n",
         program.size, minimind2_small::kContextLength);
#else
    info("MiniMind2 PTE bytes=%zu; greedy generation, context=%zu, max_new_tokens=%zu\n",
         program.size,
         minimind2_small::kContextLength,
         std::size(minimind2_small::kGeneratedExpected));
#endif
    info("MINIMIND_CLOCK cpu_hz=%" PRIu32 " npu_hz=%u npu_clock_source=configured_FPGA_assumption\n",
         get_mps4_core_clock(),
         static_cast<unsigned>(MINIMIND_NPU_CLOCK_HZ));

#ifdef MINIMIND_DEMO
    minimind2_small::RunDemo(model);
    return;
#endif
    minimind2_small::Timing promptTiming{}, decodeTiming{};
    size_t promptMatches = 0;
    int32_t nextToken    = -1;
    for (size_t position = 0; position < std::size(minimind2_small::kPromptTokens);
         ++position) {
        const int32_t token = minimind2_small::kPromptTokens[position];
        if (!minimind2_small::Infer(
                model, token, position, nextToken, promptTiming, "prompt")) {
            return;
        }
        const int32_t expected = minimind2_small::kPromptExpected[position];
        promptMatches += nextToken == expected;
        info("MINIMIND position=%zu input=%" PRId32 " top1=%" PRId32 " expected=%" PRId32
             " match=%d\n",
             position,
             token,
             nextToken,
             expected,
             nextToken == expected);
        if (position <= 1) {
#ifndef MINIMIND_INT16_KV
            minimind2_small::DumpCaches(position == 0 ? "p0" : "p1");
#endif
        }
    }
    info("MINIMIND_SMOKE matches=%zu total=%zu success=%d last_expected=%" PRId32
         " last_actual=%" PRId32 " reference=host_quantized_PT2\n",
         promptMatches,
         std::size(minimind2_small::kPromptTokens),
         promptMatches == std::size(minimind2_small::kPromptTokens),
         minimind2_small::kPromptExpected[std::size(minimind2_small::kPromptTokens) - 1],
         nextToken);

    int32_t generated[std::size(minimind2_small::kGeneratedExpected)]{};
    size_t generatedCount    = 0;
    size_t generationMatches = 0;
    for (size_t index = 0; index < std::size(generated); ++index) {
        generated[generatedCount++] = nextToken;
        const int32_t expected      = minimind2_small::kGeneratedExpected[index];
        generationMatches += nextToken == expected;
        info("MINIMIND_GENERATED index=%zu token=%" PRId32 " expected=%" PRId32
             " match=%d\n",
             index,
             nextToken,
             expected,
             nextToken == expected);
        if (nextToken == 2 || index + 1 == std::size(generated)) {
            break;
        }
        const size_t position = std::size(minimind2_small::kPromptTokens) + index;
        if (position >= minimind2_small::kContextLength ||
            !minimind2_small::Infer(
                model, nextToken, position, nextToken, decodeTiming, "decode")) {
            return;
        }
    }
    info("MINIMIND_GENERATION matches=%zu count=%zu success=%d\n",
         generationMatches,
         generatedCount,
         generatedCount == std::size(generated) &&
             generationMatches == generatedCount);
    static char textBuffer[4096];
    size_t textBytes = 0;
    if (!minimind2_small::DecodeTokenIds(generated,
                                         generatedCount,
                                         textBuffer,
                                         sizeof(textBuffer),
                                         textBytes)) {
        printf_err("MiniMind2 text decoding failed\n");
    } else {
        info("MINIMIND_TEXT %s\n", textBuffer);
    }

    // Zero-cache position-1 probe: isolate the two-valid softmax from the
    // position-0 cache attention. Repeating it from an identical state checks
    // whether the delegate is stateless between invocations.
#ifndef MINIMIND_NO_CACHE_RESET
    minimind2_small::ResetCaches();
    int32_t zeroToken = -1;
    minimind2_small::Timing zeroTiming{};
    if (!minimind2_small::Infer(model, 85, 1, zeroToken, zeroTiming, "zero1")) {
        printf_err("MiniMind2 zero-cache probe failed\n");
    }
    minimind2_small::ResetCaches();
    int32_t repeatToken = -1;
    minimind2_small::Timing repeatTiming{};
    if (!minimind2_small::Infer(model, 85, 1, repeatToken, repeatTiming, "zero2")) {
        printf_err("MiniMind2 repeat probe failed\n");
    }
    info("MINIMIND_ZERO1 top1=%" PRId32 " repeat=%" PRId32 "\n", zeroToken, repeatToken);
#endif
    minimind2_small::PrintTiming("prompt_sum", 0, promptTiming);
    minimind2_small::PrintTiming("decode_sum", 0, decodeTiming);
#ifdef MINIMIND_PROFILE_STAGES
    minimind2_small::GetProfiler().Print();
#endif
    model.LogMemoryUsage();
}
