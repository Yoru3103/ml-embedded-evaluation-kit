/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace arm::app::tinystories {

/** Preserve Q(DQ(q)) at the one exceptional signed INT16 endpoint. */
inline void ClampCache(int16_t* data, size_t count)
{
    for (size_t index = 0; index < count; ++index) {
        if (data[index] == std::numeric_limits<int16_t>::min()) {
            data[index] = -32767;
        }
    }
}

} // namespace arm::app::tinystories
