/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#if ASR_ZH_PROFILE
#    include "AsrZhOpProfiler.hpp"
#    include "ethosu_driver.h"
#    include "hal.h"
#    include "mlek/log/log_macros.h"
#    include <cinttypes>

namespace arm::app::asr_zh {
AsrZhOpProfiler* g_profiler = nullptr;

uint64_t ReadCycles()
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

bool AsrZhOpProfiler::PrintStage(const char* name, uint64_t cycles)
{
    const uint64_t frequency = get_mps4_core_clock();
    if (!frequency) {
        printf_err("ASR_ZH timing requires a nonzero CPU clock\n");
        return false;
    }
    const uint64_t ns =
        (cycles / frequency) * 1000000000ULL + (cycles % frequency) * 1000000000ULL / frequency;
    info("ASR_ZH_STAGE %-14s cycles=%" PRIu64 " us=%" PRIu64 ".%03" PRIu64 "\n",
         name,
         cycles,
         ns / 1000,
         ns % 1000);
    return true;
}

bool AsrZhOpProfiler::Validate(uint64_t invoke) const
{
    const auto& quantize    = Get(Operation::QUANTIZE);
    const auto& npu         = Get(Operation::ETHOS_U);
    const auto& dequantize  = Get(Operation::DEQUANTIZE);
    const uint64_t measured = quantize.cycles + npu.cycles + dequantize.cycles;
    if (quantize.calls != m_expectedConversions || npu.calls != 2 ||
        dequantize.calls != m_expectedConversions || measured > invoke) {
        printf_err("ASR_ZH invalid timing: Q=%u NPU=%u DQ=%u measured=%" PRIu64 " invoke=%" PRIu64
                   "\n",
                   quantize.calls,
                   npu.calls,
                   dequantize.calls,
                   measured,
                   invoke);
        return false;
    }
    return true;
}

bool AsrZhOpProfiler::Print(size_t chunk,
                            size_t chunks,
                            uint64_t pre,
                            uint64_t invoke,
                            uint64_t post,
                            uint64_t stateCopy) const
{
    if (!Validate(invoke)) {
        return false;
    }
    const auto& quantize    = Get(Operation::QUANTIZE);
    const auto& npu         = Get(Operation::ETHOS_U);
    const auto& dequantize  = Get(Operation::DEQUANTIZE);
    const uint64_t measured = quantize.cycles + npu.cycles + dequantize.cycles;
    info("ASR_ZH timing: chunk=%zu/%zu cpu_hz=%" PRIu32 "\n",
         chunk + 1,
         chunks,
         get_mps4_core_clock());
    const char* names[]     = {"Preprocess",
                               "Quantize",
                               "NPU.Driver",
                               "Dequantize",
                               "Invoke.Other",
                               "Postprocess",
                               "State.Copy",
                               "Invoke.Total",
                               "Pipeline.Total"};
    const uint64_t cycles[] = {pre,
                               quantize.cycles,
                               npu.cycles,
                               dequantize.cycles,
                               invoke - measured,
                               post,
                               stateCopy,
                               invoke,
                               pre + invoke + post + stateCopy};
    for (size_t i = 0; i < sizeof(cycles) / sizeof(cycles[0]); ++i) {
        if (!PrintStage(names[i], cycles[i])) {
            return false;
        }
    }
    info("ASR_ZH_STAGE calls quantize=%u npu=%u dequantize=%u\n",
         quantize.calls,
         npu.calls,
         dequantize.calls);
    return true;
}
} // namespace arm::app::asr_zh

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
    auto* profiler       = arm::app::asr_zh::AsrZhOpProfiler::s_active;
    const uint64_t begin = profiler ? arm::app::asr_zh::ReadCycles() : 0;
    const int status =
        __real_ethosu_invoke_v3(driver, commands, commandSize, bases, sizes, count, argument);
    if (profiler) {
        profiler->RecordNpu(arm::app::asr_zh::ReadCycles() - begin);
    }
    return status;
}

#endif
