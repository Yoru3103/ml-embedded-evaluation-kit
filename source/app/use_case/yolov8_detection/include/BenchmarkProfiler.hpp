/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef YOLOV8_BENCHMARK_PROFILER_HPP
#define YOLOV8_BENCHMARK_PROFILER_HPP

#if YOLOV8_PROFILE_STAGES && defined(MLEK_FWK_TFLM)
#    include "tensorflow/lite/micro/micro_profiler_interface.h"

#    include <array>
#    include <cstdint>
#    include <cstring>

namespace arm::app {

/** Collect only graph operator cycles; runtime and platform timer stay outside the library. */
class BenchmarkOpProfiler : public tflite::MicroProfilerInterface {
public:
    using CycleReader = uint64_t (*)();
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

    uint32_t BeginEvent(const char* tag) override
    {
        const uint32_t ignored = static_cast<uint32_t>(Operation::COUNT);
        if (!m_active || !tag) {
            return ignored;
        }
        Operation operation = Operation::COUNT;
        if (std::strcmp(tag, "QUANTIZE") == 0) {
            operation = Operation::QUANTIZE;
        } else if (std::strcmp(tag, "ethos-u") == 0) {
            operation = Operation::ETHOS_U;
        } else if (std::strcmp(tag, "DEQUANTIZE") == 0) {
            operation = Operation::DEQUANTIZE;
        }
        const auto index = static_cast<uint32_t>(operation);
        if (index != ignored) {
            m_starts[index] = m_reader();
        }
        return index;
    }

    void EndEvent(uint32_t handle) override
    {
        if (!m_active || handle >= m_measurements.size()) {
            return;
        }
        const uint64_t finish = m_reader();
        m_measurements[handle].cycles += finish - m_starts[handle];
        ++m_measurements[handle].calls;
    }

    const Measurement& Get(Operation operation) const
    { return m_measurements[static_cast<uint32_t>(operation)]; }

private:
    CycleReader m_reader;
    bool m_active{false};
    std::array<uint64_t, static_cast<uint32_t>(Operation::COUNT)> m_starts{};
    std::array<Measurement, static_cast<uint32_t>(Operation::COUNT)> m_measurements{};
};

/** Read the CPU counter without resetting CPU or NPU profiling state. */
uint64_t ReadBenchmarkCpuCycles();

} // namespace arm::app
#elif YOLOV8_PROFILE_STAGES && defined(MLEK_FWK_EXECUTORCH)
#    include "EtBenchmarkProfiler.hpp"
#endif
#endif
