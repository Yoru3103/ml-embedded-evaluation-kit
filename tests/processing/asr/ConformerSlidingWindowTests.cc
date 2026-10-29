/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ConformerSlidingWindow.hpp"
#include <stdexcept>

using namespace arm::app::conformer_window;

static void Require(bool condition)
{
    if (!condition) {
        throw std::runtime_error("Conformer sliding-window regression");
    }
}

int main()
{
    Require(ValidOutputRows(kWindowSamples) == 174);
    Require(ValidOutputRows(512 + 6 * 160) == 1);
    Require(ValidOutputRows(512) == 0);
    /* Every retained row must cover the global subsampling grid exactly once,
     * including one sample over the old limit and partial final STFT frames. */
    for (size_t samples = kWindowSamples + 1; samples <= 480000; ++samples) {
        size_t nextRow = 0;
        for (size_t offset = 0;; offset += kHopSamples) {
            const auto window = GetWindow(samples, offset);
            Require(window.samples >= 512 && window.samples <= kWindowSamples);
            Require(window.firstRow <= window.endRow && window.endRow <= kOutputRows);
            Require(offset / (4 * 160) + window.firstRow == nextRow);
            nextRow += window.endRow - window.firstRow;
            if (window.last) {
                Require(offset + window.samples == samples);
                Require(nextRow == ValidOutputRows(samples));
                break;
            }
        }
    }
    const std::vector<std::string> labels{"▁go", "od", "▁home"};
    Decoder decoder(labels);
    /* Cropping excludes spurious edge tokens; identical tokens across a seam
     * collapse, while an intervening blank preserves a genuine repetition. */
    const float first[]{0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0};
    const float second[]{0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    Require(decoder.Append(first, 3, 4, 1, 3));
    Require(decoder.Append(second, 4, 4, 0, 4));
    Require(decoder.Result() == "goodod home");
    Require(!decoder.Append(second, 4, 4, 0, 5));
    Require(!decoder.Append(second, 4, 3, 0, 4));
    Decoder independent(labels);
    Require(independent.Append(second, 4, 4, 3, 4));
    Require(independent.Result() == "home");
}
