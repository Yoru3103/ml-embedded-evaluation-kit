/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ASR_UART_PROTOCOL_HPP
#define ASR_UART_PROTOCOL_HPP

#include <array>
#include <cstddef>
#include <cstdint>

namespace arm::app::asr_serial {
constexpr uint32_t kMaxSamples = 30 * 16000;
constexpr size_t kBlockBytes   = 512;
constexpr uint32_t kSampleRate = 16000;

struct Request {
    uint32_t id{};
    uint32_t samples{};
    uint32_t crc{};
    bool Valid() const
    { return samples >= 512 && samples <= kMaxSamples; }
};

/** Standard CRC32/ISO-HDLC, with explicit streaming state. */
inline uint32_t UpdateCrc(uint32_t crc, const uint8_t* data, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (unsigned bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ ((crc & 1U) ? 0xEDB88320U : 0U);
        }
    }
    return crc;
}

/** Scan ASR1 + little-endian request ID, sample count and PCM CRC32. */
class HeaderParser {
public:
    bool Push(uint8_t byte)
    {
        if (m_size == m_bytes.size()) {
            m_size = 0;
        }
        constexpr std::array<uint8_t, 4> magic{'A', 'S', 'R', '1'};
        if (m_size < magic.size() && byte != magic[m_size]) {
            m_size     = byte == magic[0] ? 1 : 0;
            m_bytes[0] = magic[0];
            return false;
        }
        m_bytes[m_size++] = byte;
        return m_size == m_bytes.size();
    }
    bool Empty() const
    { return m_size == 0; }
    void Reset()
    { m_size = 0; }
    Request Get() const
    { return {Word(4), Word(8), Word(12)}; }

private:
    uint32_t Word(size_t offset) const
    {
        uint32_t result = 0;
        for (unsigned i = 0; i < 4; ++i) {
            result |= static_cast<uint32_t>(m_bytes[offset + i]) << (8 * i);
        }
        return result;
    }
    std::array<uint8_t, 16> m_bytes{};
    size_t m_size{};
};
} // namespace arm::app::asr_serial
#endif
