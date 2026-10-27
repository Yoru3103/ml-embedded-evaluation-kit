/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ConformerOpProfiler.hpp"
#include <stdexcept>

namespace {
uint64_t s_cycles = 0;
uint64_t ReadCycles()
{ return s_cycles; }

void Require(bool condition)
{
    if (!condition) {
        throw std::runtime_error("ExecuTorch timing callback regression");
    }
}
} // namespace

int main()
{
    using Profiler = arm::app::ConformerOpProfiler;
    using Op       = Profiler::Operation;
    Profiler profiler{ReadCycles};
    profiler.end_profiling(profiler.start_profiling("native_call_quantize_per_tensor.out"));
    Require(profiler.Get(Op::QUANTIZE).calls == 0);
    profiler.Reset();
    s_cycles   = 100;
    auto outer = profiler.start_profiling("OPERATOR_CALL");
    auto quant = profiler.start_profiling("native_call_quantize_per_tensor.out");
    s_cycles   = 145;
    profiler.end_profiling(quant);
    profiler.end_profiling(outer);
    Require(profiler.Get(Op::QUANTIZE).calls == 1);
    Require(profiler.Get(Op::QUANTIZE).cycles == 45);
    auto dequant = profiler.start_profiling("native_call_dequantize_per_tensor.out");
    s_cycles     = 152;
    profiler.end_profiling(dequant);
    profiler.RecordNpu(300);
    Require(profiler.Get(Op::DEQUANTIZE).cycles == 7);
    Require(profiler.Get(Op::ETHOS_U).cycles == 300);
    profiler.Stop();
    Require(Profiler::s_active == nullptr);
    profiler.RecordNpu(999);
    Require(profiler.Get(Op::ETHOS_U).calls == 1);
    profiler.Reset();
    Require(profiler.Get(Op::QUANTIZE).calls == 0);
    Require(profiler.Get(Op::DEQUANTIZE).cycles == 0);
    profiler.Stop();
}
