/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#if defined(MLEK_FWK_TFLM) && defined(ASR_SIMPLE_STAGE_PROFILING)
#    include "AsrSimpleStageProfiler.hpp"
#    include "ethosu_driver.h"

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
                                       void* userArg)
{
    const auto start = arm::app::AsrSimpleStageProfiler::ReadCycles();
    const int result =
        __real_ethosu_invoke_v3(driver, commands, commandSize, bases, sizes, count, userArg);
    arm::app::AsrSimpleStageProfiler::RecordNpu(arm::app::AsrSimpleStageProfiler::ReadCycles() -
                                                start);
    return result;
}
#endif
