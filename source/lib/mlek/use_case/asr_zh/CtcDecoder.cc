/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/asr_zh/CtcDecoder.hpp"
#include <algorithm>
#include <cmath>

namespace arm::app::asr_zh {
namespace {
    /** Reject overlong sequences, surrogates, invalid continuation and values above U+10FFFF. */
    bool ValidCharacter(const std::string& bytes, size_t first, size_t count)
    {
        const auto lead = static_cast<unsigned char>(bytes[first]);
        if (count == 1) {
            return lead < 0x80;
        }
        uint32_t value = 0;
        if (count == 2 && lead >= 0xc2 && lead <= 0xdf) {
            value = lead & 0x1f;
        } else if (count == 3 && lead >= 0xe0 && lead <= 0xef) {
            value = lead & 0x0f;
        } else if (count == 4 && lead >= 0xf0 && lead <= 0xf4) {
            value = lead & 0x07;
        } else {
            return false;
        }
        for (size_t i = 1; i < count; ++i) {
            const auto next = static_cast<unsigned char>(bytes[first + i]);
            if ((next & 0xc0) != 0x80) {
                return false;
            }
            value = (value << 6) | (next & 0x3f);
        }
        const uint32_t minimum = count == 2 ? 0x80 : (count == 3 ? 0x800 : 0x10000);
        return value >= minimum && value <= 0x10ffff && !(value >= 0xd800 && value <= 0xdfff);
    }
} // namespace

void CtcDecoder::Reset()
{
    m_previous = -1;
    m_bytes.clear();
    m_path.clear();
}

bool CtcDecoder::AppendToken(size_t token)
{
    if (!m_tokens || token >= m_count) {
        return false;
    }
    if (token != 0 && static_cast<int32_t>(token) != m_previous) {
        const auto& piece = m_tokens[token];
        if (piece.size != 0 && !piece.data) {
            return false;
        }
        if (piece.size) {
            m_bytes.append(reinterpret_cast<const char*>(piece.data), piece.size);
        }
    }
    m_previous = static_cast<int32_t>(token);
    m_path.push_back(static_cast<uint32_t>(token));
    return true;
}

bool CtcDecoder::Append(const float* scores, size_t frames, size_t classes)
{
    if (!scores || classes != m_count || frames == 0) {
        return false;
    }
    for (size_t row = 0; row < frames; ++row) {
        size_t best = 0;
        for (size_t column = 0; column < classes; ++column) {
            if (!std::isfinite(scores[row * classes + column])) {
                return false;
            }
            if (scores[row * classes + column] > scores[row * classes + best]) {
                best = column;
            }
        }
        if (!AppendToken(best)) {
            return false;
        }
    }
    return true;
}

std::string CtcDecoder::Text() const
{
    std::vector<size_t> counts(m_bytes.size() + 1, 0);
    std::vector<size_t> previous(m_bytes.size() + 1, 0);
    for (size_t end = 1; end <= m_bytes.size(); ++end) {
        counts[end]   = counts[end - 1];
        previous[end] = end - 1;
        for (size_t size = 1; size <= std::min<size_t>(4, end); ++size) {
            if (counts[end - size] + 1 > counts[end] && ValidCharacter(m_bytes, end - size, size)) {
                counts[end]   = counts[end - size] + 1;
                previous[end] = end - size;
            }
        }
    }
    std::vector<std::string> characters;
    for (size_t end = m_bytes.size(); end > 0; end = previous[end]) {
        if (counts[end] == counts[previous[end]] + 1) {
            characters.push_back(m_bytes.substr(previous[end], end - previous[end]));
        }
    }
    std::string result;
    for (auto it = characters.rbegin(); it != characters.rend(); ++it) {
        if (*it != " ") {
            result += *it;
        }
    }
    return result;
}
} // namespace arm::app::asr_zh
