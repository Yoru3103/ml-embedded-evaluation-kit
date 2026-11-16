/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#if defined(MLEK_FWK_EXECUTORCH) && defined(ASR_ET_STAGE_PROFILING)
#    include "ConformerOpProfiler.hpp"
#    include "ethosu_driver.h"
#    include "hal.h"
#    include "mlek/log/log_macros.h"
#    include <cinttypes>

namespace arm::app {
uint64_t ReadConformerCpuCycles()
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

bool ConformerOpProfiler::Print(uint64_t pre, uint64_t invoke, uint64_t post) const
{
    const auto& quant       = Get(Operation::QUANTIZE);
    const auto& npu         = Get(Operation::ETHOS_U);
    const auto& dequant     = Get(Operation::DEQUANTIZE);
    const uint64_t measured = quant.cycles + npu.cycles + dequant.cycles;
    if (quant.calls != 1 || npu.calls != 1 || dequant.calls != 1 || measured > invoke) {
        printf_err("ASR timing invalid: Q=%" PRIu32 ", NPU=%" PRIu32 ", DQ=%" PRIu32
                   ", measured=%" PRIu64 ", invoke=%" PRIu64 "\n",
                   quant.calls,
                   npu.calls,
                   dequant.calls,
                   measured,
                   invoke);
        return false;
    }
    const uint64_t other    = invoke - measured;
    const uint64_t total    = pre + invoke + post;
    const char* names[]     = {"Preprocess",
                               "Quantize",
                               "NPU.Driver",
                               "Dequantize",
                               "Invoke.Other",
                               "Postprocess",
                               "Invoke.Total",
                               "Pipeline.Total"};
    const uint64_t cycles[] = {
        pre, quant.cycles, npu.cycles, dequant.cycles, other, post, invoke, total};
    const uint64_t frequency = get_mps4_core_clock();
    if (frequency == 0) {
        printf_err("ASR timing requires a nonzero CPU clock\n");
        return false;
    }
    info("ASR timing: CPU elapsed cycles; NPU.Driver includes submission/wait; "
         "display/audio/UART excluded from pipeline total\n");
    for (size_t i = 0; i < sizeof(cycles) / sizeof(cycles[0]); ++i) {
        const uint64_t ns = (cycles[i] / frequency) * 1000000000ULL +
                            (cycles[i] % frequency) * 1000000000ULL / frequency;
        info("ASR_STAGE %s cycles=%" PRIu64 " us=%" PRIu64 ".%03" PRIu64 "\n",
             names[i],
             cycles[i],
             ns / 1000,
             ns % 1000);
    }
    info("ASR_STAGE calls quantize=%" PRIu32 " npu=%" PRIu32 " dequantize=%" PRIu32 "\n",
         quant.calls,
         npu.calls,
         dequant.calls);
    return true;
}
} // namespace arm::app

extern "C" int __real_ethosu_invoke_v3(struct ethosu_driver*,
                                       const void*,
                                       const int,
                                       uint64_t* const,
                                       const size_t*,
                                       const int,
                                       void*);

extern "C" int __wrap_ethosu_invoke_v3(struct ethosu_driver* drv,
                                       const void* commands,
                                       const int commandSize,
                                       uint64_t* const bases,
                                       const size_t* sizes,
                                       const int count,
                                       void* userArg)
{
    auto* profiler = arm::app::ConformerOpProfiler::s_active;
    if (!profiler) {
        return __real_ethosu_invoke_v3(drv, commands, commandSize, bases, sizes, count, userArg);
    }
    const uint64_t start = arm::app::ReadConformerCpuCycles();
    const int result =
        __real_ethosu_invoke_v3(drv, commands, commandSize, bases, sizes, count, userArg);
    profiler->RecordNpu(arm::app::ReadConformerCpuCycles() - start);
    return result;
}
#endif
