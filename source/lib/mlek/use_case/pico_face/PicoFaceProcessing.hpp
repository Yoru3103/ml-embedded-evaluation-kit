/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstddef>
#include <cstdint>

namespace arm::app::pico_face {
/** Clamp [-1,1], round to bytes, and interleave CHW RGB into HWC RGB888. */
void ConvertToRgb(const float* image, uint8_t* rgb, size_t pixels);
/** Standard reflected CRC-32, polynomial 0xedb88320. */
uint32_t ImageCrc32(const uint8_t* data, size_t size);
} // namespace arm::app::pico_face
