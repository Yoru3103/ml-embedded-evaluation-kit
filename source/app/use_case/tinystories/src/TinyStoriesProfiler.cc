/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#include "TinyStoriesProfiler.hpp"
#include "TinyStoriesProfileContract.hpp"
#include "ethosu_driver.h"
#include "mlek/log/log_macros.h"
#include <cinttypes>
#include <cstring>
#include <executorch/runtime/core/event_tracer_hooks.h>

namespace arm::app::tinystories {
bool TinyStoriesProfiler::CheckEventHooks()
{
    Reset();
    {
        executorch::runtime::internal::EventTracerProfileOpScope scope(
            this, "native_call_quantize_per_tensor.out");
    }
    Stop();
    const auto& measurement = m_measurements[static_cast<size_t>(Stage::QUANTIZE)];
    const bool valid        = measurement.calls == 1 && measurement.cycles > 0;
    info("TINYSTORIES_TRACER_SELFTEST success=%d calls=%u cycles=%" PRIu64 "\n",
         valid,
         measurement.calls,
         measurement.cycles);
    return valid;
}
void TinyStoriesProfiler::Reset()
{
    m_measurements = {};
    m_driverDone   = false;
    s_active       = this;
}
void TinyStoriesProfiler::Stop()
{ s_active = nullptr; }
void TinyStoriesProfiler::DelegateBegin()
{
    m_driverDone    = false;
    m_delegateBegin = Now();
}
void TinyStoriesProfiler::DelegateEnd()
{ Record(Stage::DELEGATE, Now() - m_delegateBegin); }
void TinyStoriesProfiler::Record(Stage stage, uint64_t cycles, size_t bytes)
{
    auto& item = m_measurements[static_cast<size_t>(stage)];
    item.cycles += cycles;
    item.bytes += bytes;
    ++item.calls;
}
TinyStoriesProfiler::Entry
TinyStoriesProfiler::start_profiling(const char* name, ChainId, DebugHandle)
{
    Entry entry{};
    entry.event_id = static_cast<int64_t>(Stage::COUNT);
    if (s_active != this || !name || std::strncmp(name, "native_call_", 12) != 0) {
        return entry;
    }
    Stage stage = Stage::NATIVE_OTHER;
    if (std::strstr(name, "dequantize_per_tensor")) {
        stage = Stage::DEQUANTIZE;
    } else if (std::strstr(name, "quantize_per_tensor")) {
        stage = Stage::QUANTIZE;
    } else if (std::strstr(name, "embedding_byte")) {
        stage = Stage::EMBEDDING;
    } else if (std::strstr(name, "index.Tensor")) {
        stage = Stage::INDEX;
    } else if (std::strstr(name, "clamp.out")) {
        stage = Stage::CACHE_CLAMP;
    } else if (std::strstr(name, "copy_")) {
        stage = Stage::STATE_COPY;
    }
    entry.event_id   = static_cast<int64_t>(stage);
    entry.start_time = Now();
    return entry;
}
void TinyStoriesProfiler::end_profiling(Entry entry)
{
    if (s_active == this && entry.event_id >= 0 &&
        entry.event_id < static_cast<int64_t>(Stage::COUNT)) {
        Record(static_cast<Stage>(entry.event_id), Now() - entry.start_time);
    }
}

bool TinyStoriesProfiler::Print(size_t position,
                                uint64_t invoke,
                                uint32_t cpuHz,
                                bool verbose) const
{
    constexpr const char* names[] = {"CPU.Embedding",
                                     "CPU.Quantize",
                                     "CPU.Dequantize",
                                     "CPU.Index",
                                     "CPU.StateCopy",
                                     "CPU.CacheClamp",
                                     "CPU.OtherKernel",
                                     "Delegate.Total",
                                     "Delegate.InputCopy",
                                     "Delegate.OutputCopy",
                                     "NPU.Driver"};
    static_assert(sizeof(kProfileExpectedCalls) / sizeof(kProfileExpectedCalls[0]) ==
                  static_cast<size_t>(Stage::COUNT));
    bool countsValid     = true;
    uint32_t nativeCalls = 0;
    for (size_t index = 0; index < static_cast<size_t>(Stage::COUNT); ++index) {
        const auto& measurement = m_measurements[index];
        if (index < static_cast<size_t>(Stage::DELEGATE)) {
            nativeCalls += measurement.calls;
        }
        if (measurement.calls != kProfileExpectedCalls[index] ||
            (measurement.calls != 0 && measurement.cycles == 0)) {
            printf_err("TINYSTORIES_STAGE invalid position=%zu stage=%zu "
                       "expected_calls=%u actual_calls=%u cycles=%" PRIu64 "\n",
                       position,
                       index,
                       kProfileExpectedCalls[index],
                       measurement.calls,
                       measurement.cycles);
            countsValid = false;
        }
    }
    if (!countsValid) {
        return false;
    }
    uint64_t accounted = 0;
    for (size_t i = 0; i <= static_cast<size_t>(Stage::DELEGATE); ++i) {
        accounted += m_measurements[i].cycles;
    }
    const auto& delegate    = m_measurements[static_cast<size_t>(Stage::DELEGATE)];
    const auto& driver      = m_measurements[static_cast<size_t>(Stage::DRIVER)];
    const uint64_t children = driver.cycles +
                              m_measurements[static_cast<size_t>(Stage::INPUT_COPY)].cycles +
                              m_measurements[static_cast<size_t>(Stage::OUTPUT_COPY)].cycles;
    if (accounted > invoke || children > delegate.cycles || driver.calls != 1 ||
        delegate.calls != 1) {
        printf_err("TINYSTORIES_STAGE invalid position=%zu driver_calls=%u delegate_calls=%u\n",
                   position,
                   driver.calls,
                   delegate.calls);
        return false;
    }
    if (!verbose) {
        return true;
    }
    for (size_t i = 0; i < static_cast<size_t>(Stage::COUNT); ++i) {
        const auto& item = m_measurements[i];
        info("TINYSTORIES_STAGE position=%zu name=%s calls=%u cycles=%" PRIu64
             " ms=%.3f bytes=%" PRIu64 "\n",
             position,
             names[i],
             item.calls,
             item.cycles,
             static_cast<double>(item.cycles) * 1000.0 / cpuHz,
             item.bytes);
    }
    info("TINYSTORIES_STAGE position=%zu name=Delegate.Other cycles=%" PRIu64 " ms=%.3f\n",
         position,
         delegate.cycles - children,
         static_cast<double>(delegate.cycles - children) * 1000.0 / cpuHz);
    info("TINYSTORIES_STAGE position=%zu name=Runtime.Other cycles=%" PRIu64 " ms=%.3f\n",
         position,
         invoke - accounted,
         static_cast<double>(invoke - accounted) * 1000.0 / cpuHz);
    info("TINYSTORIES_STAGE_VALID position=%zu native_calls=%u\n", position, nativeCalls);
    return true;
}
} // namespace arm::app::tinystories

extern "C" void EthosUBackend_execute_begin()
{
    if (auto* profiler = arm::app::tinystories::TinyStoriesProfiler::s_active) {
        profiler->DelegateBegin();
    }
}
extern "C" void EthosUBackend_execute_end()
{
    if (auto* profiler = arm::app::tinystories::TinyStoriesProfiler::s_active) {
        profiler->DelegateEnd();
    }
}
extern "C" int __real_ethosu_invoke_v3(struct ethosu_driver*,
                                       const void*,
                                       const int,
                                       uint64_t* const,
                                       const size_t*,
                                       const int,
                                       void*);
extern "C" int __wrap_ethosu_invoke_v3(struct ethosu_driver* driver,
                                       const void* commands,
                                       const int commandSize,
                                       uint64_t* const bases,
                                       const size_t* sizes,
                                       const int count,
                                       void* argument)
{
    auto* profiler       = arm::app::tinystories::TinyStoriesProfiler::s_active;
    const uint64_t begin = profiler ? profiler->Now() : 0;
    const int result =
        __real_ethosu_invoke_v3(driver, commands, commandSize, bases, sizes, count, argument);
    if (profiler) {
        profiler->Record(arm::app::tinystories::TinyStoriesProfiler::Stage::DRIVER,
                         profiler->Now() - begin);
        profiler->SetDriverDone();
    }
    return result;
}
extern "C" void __real_arm_ethos_io_memcpy(void*, const void*, size_t);
extern "C" void __wrap_arm_ethos_io_memcpy(void* destination, const void* source, size_t size)
{
    auto* profiler       = arm::app::tinystories::TinyStoriesProfiler::s_active;
    const uint64_t begin = profiler ? profiler->Now() : 0;
    __real_arm_ethos_io_memcpy(destination, source, size);
    if (profiler) {
        using Stage = arm::app::tinystories::TinyStoriesProfiler::Stage;
        profiler->Record(profiler->DriverDone() ? Stage::OUTPUT_COPY : Stage::INPUT_COPY,
                         profiler->Now() - begin,
                         size);
    }
}
