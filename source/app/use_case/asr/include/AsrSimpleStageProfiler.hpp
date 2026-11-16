/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ASR_SIMPLE_STAGE_PROFILER_HPP
#define ASR_SIMPLE_STAGE_PROFILER_HPP

#if defined(ASR_SIMPLE_STAGE_PROFILING)
#    include "AsrSimpleStageTimings.hpp"
#    include "hal.h"
#    include "mlek/common/ProcessingStageScope.hpp"
#    include "mlek/log/log_macros.h"
#    include <cstring>

namespace arm::app {

/** CPU-clock elapsed timings. No UART, allocations or per-frame hooks during processing. */
class AsrSimpleStageProfiler {
public:
    using Stage = AsrSimpleStageTimings::Stage;

    AsrSimpleStageProfiler()
    {
        hal_pmu_reset();
        hal_pmu_final();
        ProcessingStageScope::s_observer = Observe;
    }

    ~AsrSimpleStageProfiler()
    { ProcessingStageScope::s_observer = nullptr; }

    static uint64_t ReadCycles()
    {
        pmu_counters counters{};
        hal_pmu_get_cpu_counters(&counters);
        for (uint32_t i = 0; i < counters.num_counters; ++i) {
            if (std::strcmp(counters.counters[i].name, "CPU TOTAL") == 0) {
                return counters.counters[i].value;
            }
        }
        return 0;
    }

    static void RecordNpu(uint64_t cycles)
    {
        if (s_invoking) {
            s_timings.Record(Stage::NPU, cycles);
        }
    }

    static bool PrintAndReset()
    {
        AsrSimpleStageTimings::Durations values{};
        const uint64_t frequency = get_mps4_core_clock();
        if (s_invoking || !s_timings.Calculate(values) || frequency == 0) {
            printf_err("Invalid wav2letter stage timings; expected one call per stage\n");
            return false;
        }
        const char* names[] = {
            "Preprocess", "Quantize", "NPU.Driver", "Dequantize", "Invoke.Other", "Postprocess"};
        for (size_t i = 0; i < values.size(); ++i) {
            const uint64_t ns = (values[i] / frequency) * 1000000000ULL +
                                (values[i] % frequency) * 1000000000ULL / frequency;
            info("Profile for %s:\n", names[i]);
            info("CPU TOTAL: %" PRIu64 " cycles\n", values[i]);
            info("DURATION: %" PRIu64 ".%03" PRIu64 " us\n", ns / 1000, ns % 1000);
        }
        s_timings = {};
        return true;
    }

private:
    inline static AsrSimpleStageTimings s_timings{};
    inline static std::array<uint64_t, static_cast<size_t>(Stage::COUNT)> s_starts{};
    inline static bool s_invoking = false;

    static void Observe(const char* name, bool begin)
    {
        Stage stage = Stage::COUNT;
        if (std::strcmp(name, "Preprocess") == 0) {
            stage = Stage::PREPROCESS;
        } else if (std::strcmp(name, "Quantize") == 0) {
            stage = Stage::QUANTIZE;
        } else if (std::strcmp(name, "Dequantize") == 0) {
            stage = Stage::DEQUANTIZE;
        } else if (std::strcmp(name, "Postprocess") == 0) {
            stage = Stage::POSTPROCESS;
        } else if (std::strcmp(name, "Invoke") == 0) {
            stage = Stage::INVOKE;
        } else {
            return;
        }
        const auto now   = ReadCycles();
        const auto index = static_cast<size_t>(stage);
        if (begin) {
            s_starts[index] = now;
        } else {
            s_timings.Record(stage, now - s_starts[index]);
        }
        if (stage == Stage::INVOKE) {
            s_invoking = begin;
        }
    }
};
} // namespace arm::app
#endif
#endif
