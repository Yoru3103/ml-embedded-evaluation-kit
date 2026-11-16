/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <array>
#include <cstdint>
#include <executorch/runtime/core/event_tracer.h>
#ifndef ET_EVENT_TRACER_ENABLED
#    error "TinyStories profiling requires ExecuTorch runtime event callbacks"
#endif
namespace arm::app::tinystories {
class TinyStoriesProfiler : public executorch::runtime::EventTracer {
public:
    using CycleReader = uint64_t (*)();
    using Entry       = executorch::runtime::EventTracerEntry;
    using ChainId     = executorch::runtime::ChainID;
    using DebugHandle = executorch::runtime::DebugHandle;
    using DelegateId  = executorch::runtime::DelegateDebugIntId;
    using AllocatorId = executorch::runtime::AllocatorID;
    using EValue      = executorch::runtime::EValue;
    using LoggedType  = executorch::runtime::LoggedEValueType;
    using Filter      = executorch::runtime::EventTracerFilterBase;
    using Tensor      = executorch::aten::Tensor;
    using TensorArray = executorch::runtime::ArrayRef<Tensor>;
    using Result      = executorch::runtime::Result<bool>;
    enum class Stage : size_t {
        EMBEDDING,
        QUANTIZE,
        DEQUANTIZE,
        INDEX,
        STATE_COPY,
        CACHE_CLAMP,
        NATIVE_OTHER,
        DELEGATE,
        INPUT_COPY,
        OUTPUT_COPY,
        DRIVER,
        COUNT
    };
    struct Measurement {
        uint64_t cycles = 0;
        uint64_t bytes  = 0;
        uint32_t calls  = 0;
    };
    using Snapshot = std::array<Measurement, static_cast<size_t>(Stage::COUNT)>;
    const Snapshot& GetSnapshot() const
    { return m_measurements; }
    void SetSnapshot(const Snapshot& snapshot)
    { m_measurements = snapshot; }
    explicit TinyStoriesProfiler(CycleReader reader) : m_reader(reader) {}
    bool CheckEventHooks();
    void Reset();
    void Stop();
    void DelegateBegin();
    void DelegateEnd();
    void Record(Stage stage, uint64_t cycles, size_t bytes = 0);
    bool Print(size_t position, uint64_t invoke, uint32_t cpuHz, bool verbose = true) const;
    uint64_t Now() const
    { return m_reader(); }
    bool DriverDone() const
    { return m_driverDone; }
    void SetDriverDone()
    { m_driverDone = true; }
    inline static TinyStoriesProfiler* s_active = nullptr;
    Entry start_profiling(const char* name, ChainId = -1, DebugHandle = 0) override;
    void end_profiling(Entry entry) override;
    void create_event_block(const char*) override {}
    Entry start_profiling_delegate(const char*, DelegateId) override
    {
        Entry entry{};
        entry.event_id = static_cast<int64_t>(Stage::COUNT);
        return entry;
    }
    void end_profiling_delegate(Entry, const void*, size_t) override {}
    void log_profiling_delegate(
        const char*, DelegateId, et_timestamp_t, et_timestamp_t, const void*, size_t) override
    {}
    void track_allocation(AllocatorId, size_t) override {}
    AllocatorId track_allocator(const char*) override
    { return 0; }
    Result log_evalue(const EValue&, LoggedType) override
    { return false; }
    Result log_intermediate_output_delegate(const char*, DelegateId, const Tensor&) override
    { return false; }
    Result log_intermediate_output_delegate(const char*, DelegateId, const TensorArray) override
    { return false; }
    Result log_intermediate_output_delegate(const char*, DelegateId, const int&) override
    { return false; }
    Result log_intermediate_output_delegate(const char*, DelegateId, const bool&) override
    { return false; }
    Result log_intermediate_output_delegate(const char*, DelegateId, const double&) override
    { return false; }
    void set_delegation_intermediate_output_filter(Filter*) override {}

private:
    CycleReader m_reader;
    bool m_driverDone        = false;
    uint64_t m_delegateBegin = 0;
    std::array<Measurement, static_cast<size_t>(Stage::COUNT)> m_measurements{};
};
} // namespace arm::app::tinystories
