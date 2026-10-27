/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "AsrSimpleStageTimings.hpp"
#include <stdexcept>

using Timings = arm::app::AsrSimpleStageTimings;
using Stage   = Timings::Stage;

static Timings MakeWindow(
    uint64_t pre, uint64_t quant, uint64_t npu, uint64_t dequant, uint64_t invoke, uint64_t post)
{
    Timings result;
    result.Record(Stage::PREPROCESS, pre);
    result.Record(Stage::QUANTIZE, quant);
    result.Record(Stage::NPU, npu);
    result.Record(Stage::DEQUANTIZE, dequant);
    result.Record(Stage::INVOKE, invoke);
    result.Record(Stage::POSTPROCESS, post);
    return result;
}

int main()
{
    Timings::Durations result{};
    auto first = MakeWindow(100, 20, 30, 3, 40, 10);
    if (!first.Calculate(result) || result != Timings::Durations{80, 20, 30, 3, 10, 7}) {
        throw std::runtime_error(
            "Six intervals must partition pre + invoke + post without overlap");
    }
    first = {};
    if (first.Calculate(result)) {
        throw std::runtime_error("A new window must not reuse preceding measurements");
    }
    auto second = MakeWindow(200, 40, 60, 6, 80, 20);
    if (!second.Calculate(result) || result != Timings::Durations{160, 40, 60, 6, 20, 14}) {
        throw std::runtime_error("The second inference must keep its own durations");
    }
    for (auto invalid : {MakeWindow(10, 11, 3, 1, 4, 2),
                         MakeWindow(10, 1, 5, 1, 4, 2),
                         MakeWindow(10, 1, 3, 3, 4, 2)}) {
        if (invalid.Calculate(result)) {
            throw std::runtime_error("Inconsistent nested intervals must not underflow");
        }
    }
    second.Record(Stage::NPU, 1);
    if (second.Calculate(result)) {
        throw std::runtime_error(
            "Unexpected multiple NPU calls must not silently change the boundary");
    }
    auto zero = MakeWindow(0, 0, 0, 0, 0, 0);
    if (!zero.Calculate(result) || result != Timings::Durations{}) {
        throw std::runtime_error("Valid zero-length intervals must be accepted");
    }
}
