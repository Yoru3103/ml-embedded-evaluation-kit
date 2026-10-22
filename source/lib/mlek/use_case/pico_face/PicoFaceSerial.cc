/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/pico_face/PicoFaceSerial.hpp"
#include <cstring>

namespace arm::app::pico_face {
std::array<uint8_t, kImageHeaderSize>
BuildImageHeader(const GenerationConfig& config, uint32_t crc, uint64_t generationUs)
{
    std::array<uint8_t, kImageHeaderSize> header{};
    std::memcpy(header.data(), "PFIMAGE1", 8);
    size_t offset    = 8;
    const auto put32 = [&](uint32_t value) {
        for (unsigned i = 0; i < 4; ++i) {
            header[offset++] = static_cast<uint8_t>(value >> (i * 8));
        }
    };
    put32(1);
    put32(kImageWidth);
    put32(kImageWidth);
    put32(3);
    put32(kImageElements);
    put32(config.seed);
    put32(config.steps);
    put32(config.classId);
    uint32_t guidance;
    static_assert(sizeof(guidance) == sizeof(config.guidance));
    std::memcpy(&guidance, &config.guidance, sizeof(guidance));
    put32(guidance);
    put32(crc);
    put32(static_cast<uint32_t>(generationUs));
    put32(static_cast<uint32_t>(generationUs >> 32));
    return header;
}
} // namespace arm::app::pico_face
