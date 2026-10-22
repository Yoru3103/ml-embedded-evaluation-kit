/*
 * SPDX-FileCopyrightText: Copyright 2022, 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef HAL_PMU_H
#define HAL_PMU_H

#include "platform_pmu.h"

/**
 * @brief   Initialise the PMU available for the platform.
 **/
void hal_pmu_init(void);

/**
 * @brief   Resets the counters.
 */
void hal_pmu_reset(void);

/** @brief Reset NPU profiling only, preserving the CPU time base. */
void hal_pmu_reset_npu(void);

/** @brief Release the NPU PMU power reference after reading counters. */
void hal_pmu_final(void);

/**
 * @brief       Gets the current counter values.
 * @param[out]  Pointer to a pmu_counters object.
 **/
void hal_pmu_get_counters(pmu_counters* counters);

/** @brief Read CPU timing counters without accessing the NPU PMU. */
void hal_pmu_get_cpu_counters(pmu_counters* counters);

#endif /* HAL_PMU_H */
