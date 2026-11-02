/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef QRCODE_IMAGE_REQUEST_HPP
#define QRCODE_IMAGE_REQUEST_HPP

#include "../../../image_demo/include/ImageRequest.hpp"

namespace arm::app::qrcode_demo {
constexpr const char* kInputFile = "qrcode-input.bin";
using image_demo::Crc32;
using image_demo::ReadLittleEndian;
using image_demo::ReadStatus;

inline ReadStatus ReadRequest(FILE* file,
                              uint32_t width,
                              uint32_t height,
                              uint8_t* image,
                              size_t capacity,
                              bool havePrevious,
                              uint32_t previous,
                              uint32_t& request)
{
    return image_demo::ReadRequest(
        file, width, height, {"QRD1", 1}, image, capacity, havePrevious, previous, request);
}
} // namespace arm::app::qrcode_demo
#endif
