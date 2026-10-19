/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef YOLOV8_ET_BENCHMARK_PROFILER_HPP
#define YOLOV8_ET_BENCHMARK_PROFILER_HPP

#include <executorch/runtime/core/event_tracer.h>

#include <array>
#include <cstdint>
#include <cstring>

#ifndef ET_EVENT_TRACER_ENABLED
#    error "ExecuTorch stage profiling requires runtime event callbacks"
#endif

namespace arm::app {

/** Collect native kernel and Ethos-U backend cycles without platform dependencies. */
class BenchmarkOpProfiler : public executorch::runtime::EventTracer {
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
    enum class Operation : uint32_t { QUANTIZE, ETHOS_U, DEQUANTIZE, COUNT };

    struct Measurement {
        uint64_t cycles{0};
        uint32_t calls{0};
    };

    explicit BenchmarkOpProfiler(CycleReader reader) : m_reader(reader) {}

    void Reset()
    {
        m_measurements = {};
        m_active       = true;
    }

    void Stop()
    { m_active = false; }

    const Measurement& Get(Operation operation) const
    { return m_measurements[static_cast<uint32_t>(operation)]; }

    Entry start_profiling(const char* name, ChainId = -1, DebugHandle = 0) override
    {
        Entry entry{};
        entry.event_id = static_cast<int64_t>(Operation::COUNT);
        if (!m_active || !name) {
            return entry;
        }
        if (std::strcmp(name, "EthosUBackend::execute()") == 0) {
            entry.event_id = static_cast<int64_t>(Operation::ETHOS_U);
        } else if (std::strcmp(name, "native_call_dequantize_per_tensor.out") == 0) {
            entry.event_id = static_cast<int64_t>(Operation::DEQUANTIZE);
        } else if (std::strcmp(name, "native_call_quantize_per_tensor.out") == 0) {
            entry.event_id = static_cast<int64_t>(Operation::QUANTIZE);
        }
        if (entry.event_id < static_cast<int64_t>(Operation::COUNT)) {
            entry.start_time = m_reader();
        }
        return entry;
    }

    void end_profiling(Entry entry) override
    {
        if (m_active && entry.event_id >= 0 &&
            entry.event_id < static_cast<int64_t>(Operation::COUNT)) {
            const uint64_t finish = m_reader();
            auto& measurement     = m_measurements[static_cast<size_t>(entry.event_id)];
            measurement.cycles += finish - entry.start_time;
            ++measurement.calls;
        }
    }

    void create_event_block(const char*) override {}
    Entry start_profiling_delegate(const char*, DelegateId) override
    {
        Entry entry{};
        entry.event_id = static_cast<int64_t>(Operation::COUNT);
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
    bool m_active{false};
    std::array<Measurement, static_cast<uint32_t>(Operation::COUNT)> m_measurements{};
};

uint64_t ReadBenchmarkCpuCycles();

} // namespace arm::app
#endif
