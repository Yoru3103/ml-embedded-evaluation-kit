/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#include "TokenDecoder.hpp"
#include <cstring>

namespace arm::app::tinystories {
namespace {
    uint32_t Read32(const uint8_t* ptr)
    {
        return static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 8) |
               (static_cast<uint32_t>(ptr[2]) << 16) | (static_cast<uint32_t>(ptr[3]) << 24);
    }

    class Utf8Writer {
    public:
        Utf8Writer(char* output, size_t capacity) : m_output(output), m_capacity(capacity) {}
        bool Feed(uint8_t byte)
        {
            if (m_pendingSize != 0) {
                bool continuation = byte >= 0x80 && byte <= 0xBF;
                if (m_pendingSize == 1) {
                    continuation &= !(m_pending[0] == 0xE0 && byte < 0xA0);
                    continuation &= !(m_pending[0] == 0xED && byte > 0x9F);
                    continuation &= !(m_pending[0] == 0xF0 && byte < 0x90);
                    continuation &= !(m_pending[0] == 0xF4 && byte > 0x8F);
                }
                if (continuation) {
                    m_pending[m_pendingSize++] = byte;
                    if (m_pendingSize == m_expected) {
                        const bool result = Append(m_pending, m_pendingSize);
                        m_pendingSize     = 0;
                        return result;
                    }
                    return true;
                }
                if (!FlushInvalid()) {
                    return false;
                }
            }
            if (byte < 0x80) {
                return Append(&byte, 1);
            }
            m_expected = byte >= 0xC2 && byte <= 0xDF   ? 2
                         : byte >= 0xE0 && byte <= 0xEF ? 3
                         : byte >= 0xF0 && byte <= 0xF4 ? 4
                                                        : 0;
            if (m_expected == 0) {
                return Replacement();
            }
            m_pending[0]  = byte;
            m_pendingSize = 1;
            return true;
        }
        bool Finish(size_t& size)
        {
            if (!FlushInvalid()) {
                return false;
            }
            m_output[m_size] = '\0';
            size             = m_size;
            return true;
        }

    private:
        bool Append(const uint8_t* bytes, size_t count)
        {
            if (m_size + count >= m_capacity) {
                return false;
            }
            std::memcpy(m_output + m_size, bytes, count);
            m_size += count;
            return true;
        }
        bool FlushInvalid()
        {
            while (m_pendingSize) {
                if (!Replacement()) {
                    return false;
                }
                --m_pendingSize;
            }
            return true;
        }
        bool Replacement()
        {
            const uint8_t replacement[] = {0xEF, 0xBF, 0xBD};
            return Append(replacement, sizeof(replacement));
        }
        char* m_output;
        size_t m_capacity;
        size_t m_size = 0;
        uint8_t m_pending[4]{};
        size_t m_pendingSize = 0;
        size_t m_expected    = 0;
    };
} // namespace

bool DecodeTokens(const int32_t* tokens, size_t count, char* output, size_t capacity, size_t& bytes)
{
    bytes = 0;
    if (!tokens || !output || capacity == 0) {
        return false;
    }
    output[0]           = '\0';
    const uint8_t* data = GetTokenizerData();
    const size_t size   = GetTokenizerSize();
    if (size < 8 || std::memcmp(data, "TSV1", 4) != 0) {
        return false;
    }
    const uint32_t vocabulary = Read32(data + 4);
    if (vocabulary != 32000 || size < 8 + (vocabulary + 1) * 4) {
        return false;
    }
    const size_t base = 8 + (vocabulary + 1) * 4;
    Utf8Writer writer(output, capacity);
    bool first = true;
    for (size_t i = 0; i < count; ++i) {
        if (tokens[i] < 0 || static_cast<uint32_t>(tokens[i]) >= vocabulary) {
            return false;
        }
        const size_t begin = Read32(data + 8 + tokens[i] * 4);
        const size_t end   = Read32(data + 8 + (tokens[i] + 1) * 4);
        if (begin >= end || end > size - base) {
            return false;
        }
        const uint8_t flag = data[base + begin];
        size_t cursor      = base + begin + 1;
        const size_t limit = base + end;
        if (cursor == limit) {
            continue; // BOS/EOS/control tokens have no surface.
        }
        if (first && flag == 1 && data[cursor] == ' ') {
            ++cursor; // Remove the tokenizer's initial dummy-prefix space once.
        }
        first = false;
        for (; cursor < limit; ++cursor) {
            if (!writer.Feed(data[cursor])) {
                return false;
            }
        }
    }
    return writer.Finish(bytes);
}
} // namespace arm::app::tinystories
