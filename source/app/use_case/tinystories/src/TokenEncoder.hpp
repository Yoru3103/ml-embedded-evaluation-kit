/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <cstddef>
#include <cstdint>
namespace arm::app::tinystories {
constexpr size_t kMaxPromptBytes = 256;
const uint8_t* GetEncoderData();
size_t GetEncoderSize();
/** Encode UTF-8 with the exported identity BPE vocabulary, including BOS. */
bool EncodePrompt(const char* text, int32_t* tokens, size_t capacity, size_t& count);
} // namespace arm::app::tinystories
