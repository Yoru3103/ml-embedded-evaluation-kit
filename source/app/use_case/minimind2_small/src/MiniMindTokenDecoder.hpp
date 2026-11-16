/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace arm::app::minimind2_small {
bool DecodeTokenIds(const int32_t* ids,
                    size_t count,
                    char* output,
                    size_t capacity,
                    size_t& written);
} // namespace arm::app::minimind2_small
