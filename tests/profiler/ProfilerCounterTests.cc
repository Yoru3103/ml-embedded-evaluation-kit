/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "Profiler.hpp"

#include <cstdlib>

#define REQUIRE(condition)  \
    do {                    \
        if (!(condition)) { \
            std::abort();   \
        }                   \
    } while (false)

namespace {
unsigned s_npuReads  = 0;
uint64_t s_cpuCycles = 0;
bool s_emptyCpu      = false;
} // namespace

void hal_pmu_reset(void)
{ s_cpuCycles = 0; }

void hal_pmu_final(void) {}

void hal_pmu_get_cpu_counters(pmu_counters* counters)
{
    *counters             = {};
    counters->initialised = true;
    if (!s_emptyCpu) {
        counters->num_counters = 2;
        counters->counters[0]  = {s_cpuCycles, "CPU TOTAL", "cycles"};
        counters->counters[1]  = {s_cpuCycles / 1000, "DURATION", "milliseconds"};
        s_cpuCycles += 1000;
    }
}

void hal_pmu_get_counters(pmu_counters* counters)
{
    ++s_npuReads;
    hal_pmu_get_cpu_counters(counters);
    counters->counters[counters->num_counters++] = {500, "NPU TOTAL", "cycles"};
}

int main()
{
    arm::app::Profiler profiler;
    REQUIRE(profiler.StartProfiling("PreProcessing", false));
    REQUIRE(profiler.StopProfiling());
    REQUIRE(s_npuReads == 0);

    // The outer CPU timer also hosts the two full Invoke snapshots.
    REQUIRE(profiler.StartProfiling("InferenceAndPostProcessing", false));
    pmu_counters start{}, inferenceEnd{}, decodeEnd{};
    hal_pmu_get_counters(&start);
    hal_pmu_get_counters(&inferenceEnd);
    hal_pmu_get_cpu_counters(&decodeEnd);
    REQUIRE(profiler.StopProfiling());
    REQUIRE(s_npuReads == 2);
    REQUIRE(profiler.RecordProfiling(start, inferenceEnd, "Inference"));
    REQUIRE(profiler.RecordProfiling(inferenceEnd, decodeEnd, "Decode", false));

    std::vector<arm::app::ProfileResult> results;
    profiler.GetAllResultsAndReset(results);
    REQUIRE(results.size() == 4);
    for (const auto& result : results) {
        REQUIRE(result.data.size() == (result.name == "Inference" ? 3 : 2));
        REQUIRE(result.data[0].total ==
                (result.name == "InferenceAndPostProcessing" ? 4000 : 1000));
        for (const auto& stat : result.data) {
            REQUIRE(result.name == "Inference" || stat.name != "NPU TOTAL");
        }
    }

    // Preserve the default full-PMU API for other use cases.
    REQUIRE(profiler.StartProfiling("Default"));
    REQUIRE(profiler.StopProfiling());
    REQUIRE(s_npuReads == 4);

    // Reject malformed snapshots without adding a partial statistics series.
    pmu_counters bad     = inferenceEnd;
    bad.counters[0].name = "OTHER";
    REQUIRE(!profiler.RecordProfiling(start, bad, "BadName"));
    bad                  = inferenceEnd;
    bad.counters[0].unit = "microseconds";
    REQUIRE(!profiler.RecordProfiling(start, bad, "BadUnit"));
    bad                   = inferenceEnd;
    bad.counters[0].value = 0;
    REQUIRE(!profiler.RecordProfiling(inferenceEnd, bad, "Backwards"));
    bad.num_counters = NUM_PMU_COUNTERS + 1;
    REQUIRE(!profiler.RecordProfiling(bad, bad, "TooMany", false));

    // CPU_PROFILE_ENABLED=OFF must not index an empty statistics vector.
    s_emptyCpu = true;
    REQUIRE(profiler.StartProfiling("NoCpuCounters", false));
    REQUIRE(profiler.StopProfiling());
    results.clear();
    profiler.GetAllResultsAndReset(results);
    return 0;
}
