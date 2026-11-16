/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CONFORMER_SLIDING_WINDOW_HPP
#define CONFORMER_SLIDING_WINDOW_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace arm::app::conformer_window {
constexpr size_t kWindowSamples = 512 + 699 * 160;
constexpr size_t kHopSamples    = 496 * 160;
constexpr size_t kOutputRows    = 174;
constexpr size_t kContextRows   = 25;

struct Window {
    size_t offset;
    size_t samples;
    size_t firstRow;
    size_t endRow;
    bool last;
};

/** Two unpadded kernel-3, stride-2 convolutions reduce the feature time axis. */
inline size_t ValidOutputRows(size_t samples)
{
    const size_t frames = samples >= 512 ? 1 + (samples - 512) / 160 : 0;
    return frames >= 7 ? 1 + (frames - 7) / 4 : 0;
}

/** Adjacent retained output ranges meet on the same 40 ms grid, without overlap. */
inline Window GetWindow(size_t totalSamples, size_t offset)
{
    const size_t samples = std::min(kWindowSamples, totalSamples - offset);
    const bool last      = offset + samples == totalSamples;
    return {offset,
            samples,
            offset == 0 ? 0 : kContextRows,
            last ? ValidOutputRows(samples) : kOutputRows - kContextRows,
            last};
}

/** Collapse a single continuous CTC path before converting SentencePiece markers. */
class Decoder {
public:
    explicit Decoder(const std::vector<std::string>& labels) : m_labels(labels) {}

    bool Append(const float* logits, size_t rows, size_t columns, size_t first, size_t end)
    {
        if (!logits || columns != m_labels.size() + 1 || first > end || end > rows) {
            return false;
        }
        for (size_t row = first; row < end; ++row) {
            const auto* values = logits + row * columns;
            const auto token =
                static_cast<int32_t>(std::max_element(values, values + columns) - values);
            if (token == 0) {
                m_previous = -1;
            } else if (token != m_previous) {
                m_pieces += m_labels[static_cast<size_t>(token - 1)];
                m_previous = token;
            }
        }
        return true;
    }

    std::string Result() const
    {
        auto result              = m_pieces;
        const std::string marker = "▁";
        size_t offset            = 0;
        while ((offset = result.find(marker, offset)) != std::string::npos) {
            const std::string replacement = offset == 0 ? "" : " ";
            result.replace(offset, marker.size(), replacement);
            offset += replacement.size();
        }
        return result;
    }

private:
    const std::vector<std::string>& m_labels;
    int32_t m_previous = -1;
    std::string m_pieces;
};
} // namespace arm::app::conformer_window
#endif
