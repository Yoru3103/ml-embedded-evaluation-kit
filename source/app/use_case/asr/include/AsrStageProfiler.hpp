/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ASR_STAGE_PROFILER_HPP
#define ASR_STAGE_PROFILER_HPP

#include "mlek/common/ProcessingStageScope.hpp"

#if defined(ASR_STAGE_PROFILING)
#    include "hal.h"
#    include "mlek/log/log_macros.h"
#    include <array>
#    include <cinttypes>
#    include <cstring>

namespace arm::app {

/** Record inclusive intervals in execution order; defer UART until outside timed scopes.
 * Nested intervals must not be summed with their parents. Score dequantization is
 * recorded per output row without changing the classifier's loop structure.
 */
class AsrStageProfiler {
public:
    AsrStageProfiler()
    {
        hal_pmu_reset();
        hal_pmu_final();
        ProcessingStageScope::s_observer = Observe;
    }

    ~AsrStageProfiler()
    { ProcessingStageScope::s_observer = nullptr; }

    static void PrintAndReset()
    {
        if (s_depth != 0) {
            printf_err("Cannot print ASR timings inside a measured stage\n");
            return;
        }
        for (size_t i = 0; i < s_count; ++i) {
            const auto& event = s_events[i];
            if (event.frame != 0) {
                info("Profile for %s (frame %zu):\n", event.name, event.frame);
            } else {
                info("Profile for %s:\n", event.name);
            }
            info("CPU TOTAL: %" PRIu64 " cycles\n", event.cycles);
#    if defined(TIMER_MPS4_H)
            const auto frequency = get_mps4_core_clock();
            if (frequency != 0) {
                const uint64_t nanoseconds = event.cycles * 1000000000ULL / frequency;
                info("DURATION: %" PRIu64 ".%03" PRIu64 " us\n",
                     nanoseconds / 1000,
                     nanoseconds % 1000);
            }
#    endif
        }
        if (s_overflow) {
            printf_err("ASR timing buffer full; some individual records were omitted\n");
        }
        s_count    = 0;
        s_frame    = 0;
        s_overflow = false;
    }

private:
    struct Event {
        const char* name;
        uint64_t cycles; // Start timestamp until the corresponding scope closes.
        size_t frame;
    };

    inline static std::array<Event, 4096> s_events{};
    inline static std::array<size_t, 32> s_stack{};
    inline static size_t s_count  = 0;
    inline static size_t s_depth  = 0;
    inline static size_t s_frame  = 0;
    inline static bool s_overflow = false;

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

    static void Observe(const char* name, bool begin)
    {
        const auto now = ReadCycles();
        if (begin) {
            const bool isFrame = std::strcmp(name, "MFCC.Frame") == 0;
            if (isFrame) {
                ++s_frame;
            }
            const auto index = s_count;
            if (index < s_events.size()) {
                s_events[s_count++] = {
                    name, now, std::strncmp(name, "MFCC.", 5) == 0 ? s_frame : 0};
            } else {
                s_overflow = true;
            }
            if (s_depth < s_stack.size()) {
                s_stack[s_depth] = index;
            } else {
                s_overflow = true;
            }
            ++s_depth;
        } else if (s_depth != 0) {
            --s_depth;
            if (s_depth < s_stack.size()) {
                const auto index = s_stack[s_depth];
                if (index < s_events.size()) {
                    s_events[index].cycles = now - s_events[index].cycles;
                }
            }
        }
    }
};
} // namespace arm::app
#endif // ASR_STAGE_PROFILING
#endif // ASR_STAGE_PROFILER_HPP
