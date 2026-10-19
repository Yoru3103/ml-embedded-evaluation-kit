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
#include "hal_pmu.h"
#include "platform_drivers.h"

#if defined(ARM_NPU)
#    include "ethosu_profiler.h"
#endif

void hal_pmu_init(void)
{ platform_reset_counters(); }

void hal_pmu_reset(void)
{ platform_reset_counters(); }

void hal_pmu_final(void)
{
#if defined(ARM_NPU)
    ethosu_pmu_final();
#endif
}

void hal_pmu_get_counters(pmu_counters* counters)
{ platform_get_counters(counters); }

void hal_pmu_get_cpu_counters(pmu_counters* counters)
{ platform_get_cpu_counters(counters); }
