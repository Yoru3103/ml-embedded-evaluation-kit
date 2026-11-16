/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated from the host quantized PT2 reference.
 */
#pragma once
#include <cstddef>
#include <cstdint>

namespace arm::app::minimind2_small {
constexpr size_t kVocabularySize = 6400;
#ifndef MINIMIND_CONTEXT_LENGTH
#define MINIMIND_CONTEXT_LENGTH 64
#endif
constexpr size_t kContextLength = MINIMIND_CONTEXT_LENGTH;
static_assert(kContextLength == 64 || kContextLength == 128,
              "MiniMind2 supports only the exported 64- and 128-token PTEs");
constexpr int32_t kPromptTokens[] = {
    1, 85, 736, 201, 59, 292, 389, 260, 3836, 1861, 501, 2, 201, 1, 320, 275, 201, 1055, 411, 333, 5565, 2504, 608, 867, 286, 2, 201, 1, 1078, 538, 501, 201
};
constexpr int32_t kPromptExpected[] = {
    5, 736, 17, 1024, 81, 54, 295, 3004, 322, 501, 16, 201, 368, 5, 301, 3, 1024, 2910, 3205, 1811, 355, 355, 368, 355, 201, 201, 1024, 5, 201, 201, 201, 397
};
constexpr int32_t kGeneratedExpected[] = {
    397, 1157, 1805, 3760, 1490, 655, 1511, 269, 1783, 1886, 1958, 2933, 270, 1237, 6307, 2706
};
} // namespace arm::app::minimind2_small
