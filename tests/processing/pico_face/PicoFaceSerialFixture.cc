/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/pico_face/PicoFaceProcessing.hpp"
#include "mlek/use_case/pico_face/PicoFaceSerial.hpp"
#include <cstdio>

int main()
{
    using namespace arm::app::pico_face;
    std::array<uint8_t, kImageElements> image{};
    for (size_t i = 0; i < image.size(); ++i) {
        image[i] = static_cast<uint8_t>(i);
    }
    const auto header = BuildImageHeader({}, ImageCrc32(image.data(), image.size()), 123456789);
    if (std::fwrite(header.data(), 1, header.size(), stdout) != header.size() ||
        std::fwrite(image.data(), 1, image.size(), stdout) != image.size()) {
        return 1;
    }
    return 0;
}
