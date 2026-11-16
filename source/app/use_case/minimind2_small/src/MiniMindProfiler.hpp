/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "MiniMindRuntime.hpp"
#include "mlek/log/log_macros.h"

#include "ethosu_driver.h"

#include <array>
#include <cinttypes>
#include <cstring>
#include <cstdio>

namespace arm::app::minimind2_small {

class MiniMindProfiler final {
public:
    void Begin(const char* phase, uint64_t begin)
    {
        m_current = &m_totals[std::strcmp(phase, "prompt") == 0 ? 0 :
                              std::strcmp(phase, "decode") == 0 ? 1 : 2];
        m_begin = begin;
        m_delegateEnd = 0;
        m_delegateCalls = 0;
    }

    void Finish(uint64_t end)
    {
        if (m_delegateCalls == 1 && m_delegateEnd <= end) {
            m_current->after += end - m_delegateEnd;
        } else {
            ++m_current->invalid;
        }
        m_current->invoke += end - m_begin;
        ++m_current->calls;
        m_current = nullptr;
    }

    void Reset() { m_totals = {}; }

    void Print(bool uart = false) const
    {
        constexpr const char* phases[] = {"prompt", "decode", "probe"};
        for (size_t i = 0; i < m_totals.size(); ++i) {
            const auto& total = m_totals[i];
            if (!total.calls) { continue; }
            PrintLine(uart, "MINIMIND_PROFILE phase=%s calls=%u invalid=%u invoke=%" PRIu64
                 " pre_driver=%" PRIu64 " driver=%" PRIu64 " post_driver=%" PRIu64 "\n",
                 phases[i], total.calls, total.invalid, total.invoke, total.before,
                 total.driver, total.after);
            for (size_t kind = 0; kind < total.conversions.size(); ++kind) {
                const auto& conversion = total.conversions[kind];
                PrintLine(uart, "MINIMIND_QDQ phase=%s op=%s calls=%u elements=%" PRIu64
                     " cpu_cycles=%" PRIu64 "\n", phases[i], kind ? "dq" : "q",
                     conversion.calls, conversion.elements, conversion.cycles);
            }
#ifdef MINIMIND_INT16_KV
            for (size_t kind = 0; kind < total.clamps.size(); ++kind) {
                const auto& clamp = total.clamps[kind];
                PrintLine(uart, "MINIMIND_CLAMP phase=%s path=%s calls=%u elements=%" PRIu64
                     " cpu_cycles=%" PRIu64 "\n", phases[i], kind ? "fast" : "fallback",
                     clamp.calls, clamp.elements, clamp.cycles);
            }
#endif
        }
    }

    void RecordDriver(uint64_t begin, uint64_t end)
    {
        if (m_current) {
            m_current->driver += end - begin;
            if (m_delegateCalls++ == 0) {
                m_current->before += begin - m_begin;
            }
            m_delegateEnd = end;
        }
    }
    void RecordConversion(bool dequantize, uint64_t begin, uint64_t end, size_t elements)
    {
        if (m_current) {
            auto& conversion = m_current->conversions[dequantize];
            conversion.cycles += end - begin;
            conversion.elements += elements;
            ++conversion.calls;
        }
    }

    void RecordClamp(bool fast, uint64_t cycles, size_t elements)
    {
        if (m_current) {
            auto& clamp = m_current->clamps[fast];
            clamp.cycles += cycles;
            clamp.elements += elements;
            ++clamp.calls;
        }
    }

private:
    template <typename... Args>
    static void PrintLine(bool uart, const char* format, Args... args)
    {
        if (uart) {
            std::printf(format, args...);
        } else {
            info(format, args...);
        }
    }

    struct Conversion {
        uint64_t cycles{};
        uint64_t elements{};
        uint32_t calls{};
    };

    struct Totals {
        std::array<Conversion, 2> conversions{};
        std::array<Conversion, 2> clamps{};
        uint64_t driver{};
        uint64_t invoke{};
        uint64_t before{};
        uint64_t after{};
        uint32_t calls{};
        uint32_t invalid{};
    };

    std::array<Totals, 3> m_totals{};
    Totals* m_current{};
    uint64_t m_begin{};
    uint64_t m_delegateEnd{};
    uint32_t m_delegateCalls{};
};

inline MiniMindProfiler& GetProfiler()
{
    static MiniMindProfiler profiler;
    return profiler;
}

} // namespace arm::app::minimind2_small
