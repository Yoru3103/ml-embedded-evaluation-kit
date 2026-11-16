/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#ifdef TINYSTORIES_DEMO
#    include "TokenEncoder.hpp"
#    include <cstring>
#    include <limits>

namespace arm::app::tinystories {
namespace {
    uint32_t Read32(const uint8_t* data)
    {
        return uint32_t{data[0]} | (uint32_t{data[1]} << 8) | (uint32_t{data[2]} << 16) |
               (uint32_t{data[3]} << 24);
    }

    class Vocabulary {
    public:
        bool Init()
        {
            m_data             = GetEncoderData();
            const size_t bytes = GetEncoderSize();
            if (!m_data || bytes < 12 || std::memcmp(m_data, "TSE1", 4) != 0 ||
                Read32(m_data + 4) != 32000) {
                return false;
            }
            m_sortedCount = Read32(m_data + 8);
            if (m_sortedCount > 32000) {
                return false;
            }
            m_scores = 12 + 32001 * 4;
            m_sorted = m_scores + 32000 * 4;
            m_bytes  = m_sorted + m_sortedCount * 4;
            m_pieces = m_bytes + 256 * 4;
            return bytes >= m_pieces && Read32(m_data + 12) == 0 &&
                   Read32(m_data + 12 + 32000 * 4) == bytes - m_pieces;
        }
        int32_t Find(const char* text, size_t size) const
        {
            size_t low = 0, high = m_sortedCount;
            while (low < high) {
                const size_t middle = low + (high - low) / 2;
                const uint32_t id   = Read32(m_data + m_sorted + middle * 4);
                if (id >= 32000) {
                    return -1;
                }
                const size_t begin  = Read32(m_data + 12 + id * 4);
                const size_t end    = Read32(m_data + 12 + (id + 1) * 4);
                const size_t length = end - begin;
                const int compare =
                    std::memcmp(m_data + m_pieces + begin, text, length < size ? length : size);
                if (compare == 0 && length == size) {
                    return static_cast<int32_t>(id);
                }
                if (compare < 0 || (compare == 0 && length < size)) {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            return -1;
        }
        float Score(int32_t id) const
        {
            const uint32_t bits = Read32(m_data + m_scores + id * 4);
            float result;
            std::memcpy(&result, &bits, sizeof(result));
            return result;
        }
        int32_t Byte(uint8_t byte) const
        { return static_cast<int32_t>(Read32(m_data + m_bytes + byte * 4)); }

    private:
        const uint8_t* m_data = nullptr;
        size_t m_scores = 0, m_sorted = 0, m_sortedCount = 0, m_bytes = 0, m_pieces = 0;
    };

    size_t Utf8Length(const char* data, size_t remaining)
    {
        const uint8_t first = static_cast<uint8_t>(data[0]);
        const size_t length = first < 0x80                     ? 1
                              : first >= 0xC2 && first <= 0xDF ? 2
                              : first >= 0xE0 && first <= 0xEF ? 3
                              : first >= 0xF0 && first <= 0xF4 ? 4
                                                               : 0;
        if (!length || length > remaining) {
            return 0;
        }
        for (size_t index = 1; index < length; ++index) {
            const auto byte = static_cast<uint8_t>(data[index]);
            if (byte < 0x80 || byte > 0xBF ||
                (index == 1 &&
                 ((first == 0xE0 && byte < 0xA0) || (first == 0xED && byte > 0x9F) ||
                  (first == 0xF0 && byte < 0x90) || (first == 0xF4 && byte > 0x8F)))) {
                return 0;
            }
        }
        return length;
    }
    struct Symbol {
        size_t start;
        size_t length;
        int32_t token;
    };
} // namespace

bool EncodePrompt(const char* text, int32_t* tokens, size_t capacity, size_t& count)
{
    count = 0;
    Vocabulary vocabulary;
    if (!text || !tokens || capacity == 0 || !vocabulary.Init()) {
        return false;
    }
    const size_t length = std::strlen(text);
    if (length == 0 || length > kMaxPromptBytes) {
        return false;
    }
    char normalized[kMaxPromptBytes * 3 + 3];
    size_t bytes        = 0;
    const char marker[] = "\xe2\x96\x81";
    std::memcpy(normalized, marker, 3);
    bytes = 3;
    for (size_t index = 0; index < length; ++index) {
        if (text[index] == ' ') {
            std::memcpy(normalized + bytes, marker, 3);
            bytes += 3;
        } else {
            normalized[bytes++] = text[index];
        }
    }
    Symbol symbols[kMaxPromptBytes + 1];
    size_t symbolCount = 0;
    for (size_t offset = 0; offset < bytes;) {
        const size_t width = Utf8Length(normalized + offset, bytes - offset);
        if (!width || symbolCount == kMaxPromptBytes + 1) {
            return false;
        }
        symbols[symbolCount++] = {offset, width, vocabulary.Find(normalized + offset, width)};
        offset += width;
    }
    while (symbolCount > 1) {
        size_t selected = symbolCount;
        int32_t merged  = -1;
        float best      = -std::numeric_limits<float>::infinity();
        for (size_t index = 0; index + 1 < symbolCount; ++index) {
            const auto& left = symbols[index];
            const int32_t candidate =
                vocabulary.Find(normalized + left.start, left.length + symbols[index + 1].length);
            if (candidate >= 0 && vocabulary.Score(candidate) > best) {
                selected = index;
                merged   = candidate;
                best     = vocabulary.Score(candidate);
            }
        }
        if (selected == symbolCount) {
            break;
        }
        symbols[selected].length += symbols[selected + 1].length;
        symbols[selected].token = merged;
        for (size_t index = selected + 1; index + 1 < symbolCount; ++index) {
            symbols[index] = symbols[index + 1];
        }
        --symbolCount;
    }
    tokens[count++] = 1; // BOS; EOS is produced by generation, not appended to the prompt.
    for (size_t index = 0; index < symbolCount; ++index) {
        const auto& symbol = symbols[index];
        if (symbol.token >= 0) {
            if (count == capacity) {
                count = 0;
                return false;
            }
            tokens[count++] = symbol.token;
        } else {
            for (size_t byte = 0; byte < symbol.length; ++byte) {
                if (count == capacity) {
                    count = 0;
                    return false;
                }
                tokens[count++] =
                    vocabulary.Byte(static_cast<uint8_t>(normalized[symbol.start + byte]));
            }
        }
    }
    return true;
}
} // namespace arm::app::tinystories
#endif
