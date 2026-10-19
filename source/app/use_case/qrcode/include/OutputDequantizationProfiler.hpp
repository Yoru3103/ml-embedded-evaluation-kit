/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef QRCODE_OUTPUT_DEQUANTIZATION_PROFILER_HPP
#define QRCODE_OUTPUT_DEQUANTIZATION_PROFILER_HPP

#include "Profiler.hpp"

#include <array>
#include <cstring>

#if defined(MLEK_FWK_TFLM)
#    include "tensorflow/lite/micro/micro_profiler_interface.h"
#elif defined(MLEK_FWK_EXECUTORCH)
#    include <executorch/runtime/core/event_tracer.h>
#    include <executorch/schema/program_generated.h>
#endif

namespace arm::app {

/** Collect four real runtime kernel intervals without resetting the outer PMU timer. */
class DequantizationIntervals {
public:
    void Reset()
    { m_count = 0; }

    uint32_t Begin()
    {
        if (m_count >= m_start.size()) {
            ++m_count;
            return 0;
        }
        m_end[m_count].initialised = false;
        hal_pmu_get_cpu_counters(&m_start[m_count]);
        return static_cast<uint32_t>(++m_count);
    }

    void End(uint32_t handle)
    {
        if (handle > 0 && handle <= m_end.size()) {
            hal_pmu_get_cpu_counters(&m_end[handle - 1]);
        }
    }

    bool Record(Profiler& profiler) const
    {
        if (m_count != m_start.size()) {
            return false;
        }
        pmu_counters zero  = m_start[0];
        pmu_counters total = zero;
        for (size_t counter = 0; counter < zero.num_counters; ++counter) {
            zero.counters[counter].value  = 0;
            total.counters[counter].value = 0;
            for (size_t head = 0; head < m_count; ++head) {
                if (!m_end[head].initialised || m_start[head].num_counters != zero.num_counters ||
                    m_end[head].num_counters != zero.num_counters ||
                    m_end[head].counters[counter].value < m_start[head].counters[counter].value) {
                    return false;
                }
                total.counters[counter].value +=
                    m_end[head].counters[counter].value - m_start[head].counters[counter].value;
            }
        }
        return profiler.RecordProfiling(zero, total, "OutputDequantization");
    }

private:
    std::array<pmu_counters, 4> m_start{};
    std::array<pmu_counters, 4> m_end{};
    size_t m_count{0};
};

#if defined(MLEK_FWK_TFLM)
class OutputDequantizationProfiler :
    public tflite::MicroProfilerInterface,
    public DequantizationIntervals {
public:
    explicit OutputDequantizationProfiler(const uint8_t*) {}
    uint32_t BeginEvent(const char* tag) override
    { return tag && std::strcmp(tag, "DEQUANTIZE") == 0 ? Begin() : 0; }
    void EndEvent(uint32_t handle) override
    { End(handle); }
};
#elif defined(MLEK_FWK_EXECUTORCH)
namespace et = executorch::runtime;

class OutputDequantizationProfiler : public et::EventTracer, public DequantizationIntervals {
public:
    explicit OutputDequantizationProfiler(const uint8_t* model) :
        m_program{executorch_flatbuffer::GetProgram(model)}
    {}

    et::EventTracerEntry start_profiling(const char* name,
                                         et::ChainID     = et::kUnsetChainId,
                                         et::DebugHandle = et::kUnsetDebugHandle) override
    {
        et::EventTracerEntry entry{};
        if (name && std::strcmp(name, "OPERATOR_CALL") == 0 && IsDequantize()) {
            entry.event_id = Begin();
        }
        return entry;
    }
    void end_profiling(et::EventTracerEntry entry) override
    { End(entry.event_id); }
    void create_event_block(const char*) override {}
    et::EventTracerEntry start_profiling_delegate(const char*, et::DelegateDebugIntId) override
    { return {}; }
    void end_profiling_delegate(et::EventTracerEntry, const void*, size_t) override {}
    void log_profiling_delegate(const char*,
                                et::DelegateDebugIntId,
                                et_timestamp_t,
                                et_timestamp_t,
                                const void*,
                                size_t) override
    {}
    void track_allocation(et::AllocatorID, size_t) override {}
    et::AllocatorID track_allocator(const char*) override
    { return 0; }
    et::Result<bool> log_evalue(const et::EValue&, et::LoggedEValueType) override
    { return false; }
    et::Result<bool> log_intermediate_output_delegate(const char*,
                                                      et::DelegateDebugIntId,
                                                      const executorch::aten::Tensor&) override
    { return false; }
    et::Result<bool> log_intermediate_output_delegate(
        const char*, et::DelegateDebugIntId, const et::ArrayRef<executorch::aten::Tensor>) override
    { return false; }
    et::Result<bool>
    log_intermediate_output_delegate(const char*, et::DelegateDebugIntId, const int&) override
    { return false; }
    et::Result<bool>
    log_intermediate_output_delegate(const char*, et::DelegateDebugIntId, const bool&) override
    { return false; }
    et::Result<bool>
    log_intermediate_output_delegate(const char*, et::DelegateDebugIntId, const double&) override
    { return false; }
    void set_delegation_intermediate_output_filter(et::EventTracerFilterBase*) override {}

private:
    bool IsDequantize()
    {
        const auto* plans = m_program->execution_plan();
        if (!plans || plans->size() == 0 || current_chain_id() < 0) {
            return false;
        }
        const auto* plan   = plans->Get(0);
        const auto* chains = plan->chains();
        if (!chains || static_cast<uint32_t>(current_chain_id()) >= chains->size()) {
            return false;
        }
        const auto* instructions = chains->Get(current_chain_id())->instructions();
        if (!instructions || current_debug_handle() >= instructions->size()) {
            return false;
        }
        const auto* call = instructions->Get(current_debug_handle())->instr_args_as_KernelCall();
        if (!call || !plan->operators() || call->op_index() < 0 ||
            static_cast<uint32_t>(call->op_index()) >= plan->operators()->size()) {
            return false;
        }
        const auto* name = plan->operators()->Get(call->op_index())->name();
        return name && std::strcmp(name->c_str(), "cortex_m::dequantize_per_tensor") == 0;
    }

    const executorch_flatbuffer::Program* m_program;
};
#endif

} /* namespace arm::app */
#endif /* QRCODE_OUTPUT_DEQUANTIZATION_PROFILER_HPP */
