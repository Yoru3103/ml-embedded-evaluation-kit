/*
 * SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com>
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

#ifndef PLATFORM_DRIVERS_H
#define PLATFORM_DRIVERS_H

#include "hal_log.h" /* Logging related helpers. */

/* Platform components */
#include "RTE_Components.h" /* For CPU related defintiions */
#include "timer_mps4.h"     /* Timer functions. */

#include <stdint.h>

/**
 * @brief   Initialises the platform components.
 * @return  0 if successful, error code otherwise.
 */
int platform_init(void);

/**
 * @brief   Teardown for platform components.
 */
void platform_release(void);

/**
 * @brief   Gets the platform name.
 * @return  Pointer to the name
 */
const char* platform_name(void);

/**
 * @brief   Reads the user buttons (bit 0: KEY0, bit 1: KEY1).
 * @return  Bit mask with a set bit for each pressed button.
 */
uint32_t platform_get_user_buttons(void);

#endif /* PLATFORM_DRIVERS_H */
