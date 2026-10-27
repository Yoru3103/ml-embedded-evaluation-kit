/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ASR_SIMPLE_STAGE_TIMINGS_HPP
#define ASR_SIMPLE_STAGE_TIMINGS_HPP

#include <array>
#include <cstddef>
#include <cstdint>

namespace arm::app {

/** One inference window, with disjoint pre/quant/driver/dequant/other/post intervals. */
class AsrSimpleStageTimings {
public:
    enum class Stage : size_t {
        PREPROCESS,
        QUANTIZE,
        NPU,
        DEQUANTIZE,
        OTHER,
        POSTPROCESS,
        INVOKE,
        COUNT
    };
    using Durations = std::array<uint64_t, 6>;

    void Record(Stage stage, uint64_t cycles)
    {
        auto& value = m_values[static_cast<size_t>(stage)];
        value.cycles += cycles;
        ++value.calls;
    }

    bool Calculate(Durations& result) const
    {
        for (const auto stage : {Stage::PREPROCESS,
                                 Stage::QUANTIZE,
                                 Stage::NPU,
                                 Stage::DEQUANTIZE,
                                 Stage::POSTPROCESS,
                                 Stage::INVOKE}) {
            if (m_values[static_cast<size_t>(stage)].calls != 1) {
                return false;
            }
        }
        const auto cycles = [this](Stage stage) {
            return m_values[static_cast<size_t>(stage)].cycles;
        };
        if (cycles(Stage::QUANTIZE) > cycles(Stage::PREPROCESS) ||
            cycles(Stage::DEQUANTIZE) > cycles(Stage::POSTPROCESS) ||
            cycles(Stage::NPU) > cycles(Stage::INVOKE)) {
            return false;
        }
        result = {cycles(Stage::PREPROCESS) - cycles(Stage::QUANTIZE),
                  cycles(Stage::QUANTIZE),
                  cycles(Stage::NPU),
                  cycles(Stage::DEQUANTIZE),
                  cycles(Stage::INVOKE) - cycles(Stage::NPU),
                  cycles(Stage::POSTPROCESS) - cycles(Stage::DEQUANTIZE)};
        return true;
    }

private:
    struct Measurement {
        uint64_t cycles = 0;
        uint32_t calls  = 0;
    };
    std::array<Measurement, static_cast<size_t>(Stage::COUNT)> m_values{};
};
} // namespace arm::app
#endif
