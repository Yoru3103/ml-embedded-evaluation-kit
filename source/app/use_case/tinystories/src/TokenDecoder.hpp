/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <cstddef>
#include <cstdint>
namespace arm::app::tinystories {
const uint8_t* GetTokenizerData();
size_t GetTokenizerSize();
bool DecodeTokens(
    const int32_t* tokens, size_t count, char* output, size_t capacity, size_t& bytes);
} // namespace arm::app::tinystories
