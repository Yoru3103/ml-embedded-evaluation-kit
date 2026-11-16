/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#include <cstddef>
#include <cstdint>

namespace arm::app::minimind2_small {
void ClampCache(int16_t* data, size_t count)
{
    for (size_t index = 0; index < count; ++index) {
        if (data[index] == INT16_MIN) {
            data[index] = -32767;
        }
    }
}
} // namespace arm::app::minimind2_small
