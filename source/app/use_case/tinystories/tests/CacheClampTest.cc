/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#include "CacheClamp.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>

int main()
{
    std::array<int16_t, 65538> values{};
    values.front() = -123;
    values.back()  = 456;
    for (int32_t index = 0; index < 65536; ++index) {
        values[index + 1] = static_cast<int16_t>(index - 32768);
    }
    arm::app::tinystories::ClampCache(values.data() + 1, 65536);
    for (int32_t index = 0; index < 65536; ++index) {
        assert(values[index + 1] == std::max(index - 32768, -32767));
    }
    assert(values.front() == -123 && values.back() == 456);
    const auto previous = values;
    arm::app::tinystories::ClampCache(values.data() + 1, 65536);
    assert(values == previous);
    arm::app::tinystories::ClampCache(values.data(), 0);
    assert(values == previous);
    std::puts("CacheClamp: 65536 values, guards, idempotence and empty range passed");
}
