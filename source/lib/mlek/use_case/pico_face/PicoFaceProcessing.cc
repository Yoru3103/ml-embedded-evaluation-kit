/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/pico_face/PicoFaceProcessing.hpp"
#include <algorithm>
#include <cmath>

namespace arm::app::pico_face {
void ConvertToRgb(const float* image, uint8_t* rgb, size_t pixels)
{
    for (size_t p = 0; p < pixels; ++p) {
        for (size_t c = 0; c < 3; ++c) {
            const float value = std::clamp(image[c * pixels + p], -1.0f, 1.0f);
            rgb[p * 3 + c]    = static_cast<uint8_t>(std::lrintf((value + 1.0f) * 127.5f));
        }
    }
}

uint32_t ImageCrc32(const uint8_t* data, size_t size)
{
    uint32_t crc = 0xffffffffU;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (unsigned b = 0; b < 8; ++b) {
            crc = (crc >> 1) ^ ((crc & 1U) ? 0xedb88320U : 0U);
        }
    }
    return crc ^ 0xffffffffU;
}
} // namespace arm::app::pico_face
