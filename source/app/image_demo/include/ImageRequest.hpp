/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MLEK_IMAGE_REQUEST_HPP
#define MLEK_IMAGE_REQUEST_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace arm::app::image_demo {
struct Format {
    const char* magic;
    uint32_t channels;
};
enum class ReadStatus { READY, UNCHANGED, ERROR };

inline uint32_t ReadLittleEndian(const uint8_t* bytes)
{
    return uint32_t{bytes[0]} | (uint32_t{bytes[1]} << 8) | (uint32_t{bytes[2]} << 16) |
           (uint32_t{bytes[3]} << 24);
}

inline uint32_t Crc32(const uint8_t* data, size_t bytes)
{
    uint32_t crc = 0xFFFFFFFFU;
    for (size_t index = 0; index < bytes; ++index) {
        crc ^= data[index];
        for (size_t bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ ((crc & 1U) ? 0xEDB88320U : 0U);
        }
    }
    return crc ^ 0xFFFFFFFFU;
}

/** Validate one atomic host request; only READY permits inference or LCD replacement. */
inline ReadStatus ReadRequest(FILE* file,
                              uint32_t width,
                              uint32_t height,
                              const Format& format,
                              uint8_t* image,
                              size_t capacity,
                              bool havePrevious,
                              uint32_t previous,
                              uint32_t& request)
{
    std::array<uint8_t, 24> header{};
    if (!file || !image || std::fread(header.data(), 1, header.size(), file) != header.size() ||
        std::memcmp(header.data(), format.magic, 4) != 0 ||
        ReadLittleEndian(header.data() + 8) != width ||
        ReadLittleEndian(header.data() + 12) != height || width == 0 || height == 0 ||
        (format.channels != 1 && format.channels != 3) ||
        static_cast<uint64_t>(width) * height * format.channels != capacity ||
        ReadLittleEndian(header.data() + 16) != capacity) {
        return ReadStatus::ERROR;
    }
    request = ReadLittleEndian(header.data() + 4);
    if (havePrevious && request == previous) {
        return ReadStatus::UNCHANGED;
    }
    if (std::fread(image, 1, capacity, file) != capacity || std::fgetc(file) != EOF ||
        std::ferror(file) || Crc32(image, capacity) != ReadLittleEndian(header.data() + 20)) {
        return ReadStatus::ERROR;
    }
    return ReadStatus::READY;
}
} // namespace arm::app::image_demo
#endif
