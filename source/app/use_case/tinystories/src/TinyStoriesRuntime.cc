/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "BufAttributes.hpp"
#include "TinyStoriesProfiler.hpp"
#include "hal.h"
#include "mlek/fwk/executorch/EtModel.hpp"
#include "mlek/log/log_macros.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>

#include "TinyStoriesRuntime.hpp"
namespace arm::app::tinystories {
TinyStoriesProfiler* s_profiler  = nullptr;
constexpr size_t kVocabularySize = 32000;

static bool SetInt32Input(fwk::et::EtModel& model, size_t index, int32_t value)
{
    const auto tensor = model.GetInputTensor(index);
    if (!tensor || tensor->Type() != fwk::iface::TensorType::INT32 ||
        tensor->GetNumElements() != 1 || tensor->GetData<int32_t>() == nullptr) {
        printf_err("TinyStories input %zu must be a scalar INT32 tensor\n", index);
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
        printf_err("TinyStories input %zu must be a scalar INT64 tensor\n", index);
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
        printf_err("TinyStories output must be a 32000-element FP32 logits tensor\n");
        return false;
    }

    const float* logits = tensor->GetData<float>();
    float maximum       = -std::numeric_limits<float>::infinity();
    size_t topIndex     = 0;
    for (size_t i = 0; i < tensor->GetNumElements(); ++i) {
        if (!std::isfinite(logits[i])) {
            printf_err("TinyStories produced a non-finite logit at %zu\n", i);
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
{ return static_cast<double>(cycles) * 1000.0 / frequency; }

void PrintTiming(const char* phase, size_t position, const Timing& timing)
{
    const uint32_t cpuHz = get_mps4_core_clock();
    info("TINYSTORIES_TIME phase=%s position=%zu calls=%zu "
         "npu_total_cycles=%" PRIu64 " npu_total_ms=%.3f "
         "invoke_cycles=%" PRIu64 " invoke_ms=%.3f "
         "e2e_cycles=%" PRIu64 " e2e_ms=%.3f\n",
         phase,
         position,
         timing.calls,
         timing.npu,
         Milliseconds(timing.npu, TINYSTORIES_NPU_CLOCK_HZ),
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
           const char* phase,
           InferenceRecord* deferred)
{
    // PMU setup and UART are excluded; token input, inference and argmax are included.
#ifdef TINYSTORIES_DEMO
    // Keep the request's SysTick clock continuous for the final wall interval.
    // NPU counters remain per-invocation, as in TEST mode.
    hal_pmu_reset_npu();
#else
    hal_pmu_reset();
#endif
    const uint64_t begin = CpuCycles();
    if (!SetInt64Input(model, 0, token) ||
        !SetInt32Input(model, 1, static_cast<int32_t>(position))) {
        hal_pmu_final();
        return false;
    }
    s_profiler->Reset();
    const uint64_t invokeBegin = CpuCycles();
    const bool invoked         = model.RunInference();
    const uint64_t invokeEnd   = CpuCycles();
    s_profiler->Stop();
    const bool valid   = invoked && GetTopToken(model, nextToken);
    const uint64_t end = CpuCycles();
    pmu_counters counters{};
    hal_pmu_get_counters(&counters);
    hal_pmu_final();
    Timing timing{Counter(counters, "NPU TOTAL"), invokeEnd - invokeBegin, end - begin, 1};
    if (deferred) {
        deferred->timing = timing;
        deferred->stages = s_profiler->GetSnapshot();
    } else {
        PrintTiming(phase, position, timing);
        if (!s_profiler->Print(position, timing.invoke, get_mps4_core_clock())) {
            return false;
        }
    }
    // Functional FVP --fast may not simulate NPU PMU cycles. Inference and CPU
    // tracing must still pass; only FPGA requires a nonzero hardware NPU count.
#ifdef TINYSTORIES_FVP
    const bool npuTimingValid = true;
#else
    const bool npuTimingValid = timing.npu != 0;
#endif
    if (!valid || !npuTimingValid || timing.endToEnd == 0) {
        printf_err("TinyStories inference/timing failed at position %zu\n", position);
        return false;
    }
    total.npu += timing.npu;
    total.invoke += timing.invoke;
    total.endToEnd += timing.endToEnd;
    ++total.calls;
    return true;
}
} // namespace arm::app::tinystories
