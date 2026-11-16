/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MLEK_UART_FILE_INPUT_HPP
#define MLEK_UART_FILE_INPUT_HPP

#include "../../image_demo/include/ImageRequest.hpp"
#include "uart_config.h"
#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>

namespace arm::app::file_demo {
/** UartGetc maps CR to LF. Preserve every binary byte in this SSE-320 adapter. */
inline uint8_t ReadRawByte()
{
    auto* uart = reinterpret_cast<volatile uint32_t*>(UART0_BASE);
    while (!(uart[1] & 2U)) {}
    return static_cast<uint8_t>(uart[0]);
}
/** Application-level binary upload. One acknowledged block at a time avoids UART overrun.
 * A disconnected partial upload requires board reset before the next transaction.
 */
inline uint32_t Receive(const char* magic,
                        uint32_t width,
                        uint32_t height,
                        uint8_t* data,
                        size_t capacity,
                        bool variableLength   = false,
                        size_t* receivedBytes = nullptr,
                        uint8_t (*readByte)() = ReadRawByte)
{
    for (;;) {
        std::array<uint8_t, 24> header{};
        size_t matched = 0;
        while (matched < header.size()) {
            const uint8_t byte = readByte();
            if (matched == 0 && byte == '?') {
                std::printf("FILE_HELLO %.4s %" PRIu32 " %" PRIu32 " %zu 512\n",
                            magic,
                            width,
                            height,
                            capacity);
                continue;
            }
            if (matched < 4 && byte != static_cast<uint8_t>(magic[matched])) {
                matched   = byte == static_cast<uint8_t>(magic[0]) ? 1 : 0;
                header[0] = static_cast<uint8_t>(magic[0]);
                continue;
            }
            header[matched++] = byte;
        }
        const uint32_t id    = image_demo::ReadLittleEndian(header.data() + 4);
        const uint32_t bytes = image_demo::ReadLittleEndian(header.data() + 16);
        const uint32_t rows  = image_demo::ReadLittleEndian(header.data() + 12);
        if (image_demo::ReadLittleEndian(header.data() + 8) != width || bytes == 0 ||
            bytes > capacity || (!variableLength && (rows != height || bytes != capacity)) ||
            (variableLength && (rows < 400 || rows > capacity / 2 || bytes != rows * 2))) {
            std::printf("FILE_ERROR %" PRIu32 " LENGTH\n", id);
            continue;
        }
        std::printf("FILE_READY %" PRIu32 "\n", id);
        for (size_t offset = 0; offset < bytes;) {
            const size_t end = std::min(offset + size_t{512}, static_cast<size_t>(bytes));
            while (offset < end) {
                data[offset++] = readByte();
            }
            std::printf("FILE_ACK %" PRIu32 " %zu\n", id, offset);
        }
        if (image_demo::Crc32(data, bytes) != image_demo::ReadLittleEndian(header.data() + 20)) {
            std::printf("FILE_ERROR %" PRIu32 " CRC\n", id);
            continue;
        }
        if (receivedBytes) {
            *receivedBytes = bytes;
        }
        return id;
    }
}
} // namespace arm::app::file_demo
#endif
