/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "AsrUartProtocol.hpp"
#include <stdexcept>
#include <vector>

using namespace arm::app::asr_serial;

static void Require(bool condition)
{
    if (!condition) {
        throw std::runtime_error("ASR UART protocol regression");
    }
}

int main()
{
    const uint8_t check[] = "123456789";
    Require((UpdateCrc(0xFFFFFFFFU, check, 9) ^ 0xFFFFFFFFU) == 0xCBF43926U);
    const auto first = UpdateCrc(0xFFFFFFFFU, check, 4);
    Require(UpdateCrc(first, check + 4, 5) == UpdateCrc(0xFFFFFFFFU, check, 9));
    /* PCM must preserve CR, LF, NUL and high-bit bytes. */
    std::vector<uint8_t> bytes{0, 13, 10, 128, 255};
    Require(UpdateCrc(0xFFFFFFFFU, bytes.data(), bytes.size()) != UpdateCrc(0xFFFFFFFFU, check, 5));
    HeaderParser parser;
    for (auto byte : {'x', 'A', 'S', 'x', 'A'}) {
        Require(!parser.Push(byte));
    }
    for (auto byte : {'S', 'R', '1'}) {
        Require(!parser.Push(byte));
    }
    const std::array<uint32_t, 3> words{42, 51360, 0xCBF43926U};
    size_t count = 0;
    for (auto word : words) {
        for (unsigned i = 0; i < 4; ++i) {
            const auto ready = parser.Push(static_cast<uint8_t>(word >> (8 * i)));
            Require(ready == (++count == 12));
        }
    }
    const auto request = parser.Get();
    Require(request.id == 42 && request.samples == 51360 && request.crc == words[2]);
    Require(request.Valid());
    Require(!Request{1, 511, 0}.Valid());
    Require(Request{1, 512, 0}.Valid());
    Require(Request{1, kMaxSamples, 0}.Valid());
    Require(!Request{1, kMaxSamples + 1, 0}.Valid());
    /* A completed parser can accept the next frame without writing past its buffer. */
    for (auto byte : {'A', 'S', 'R', '1'}) {
        Require(!parser.Push(byte));
    }
    for (auto word : words) {
        for (unsigned i = 0; i < 4; ++i) {
            parser.Push(static_cast<uint8_t>(word >> (8 * i)));
        }
    }
    Require(parser.Get().crc == words[2]);
    parser.Reset();
    Require(parser.Empty());
}
