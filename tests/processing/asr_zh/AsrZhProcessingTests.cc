/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "FbankFixture.hpp"
#include "TokenBytes.hpp"
#include "Validation.hpp"
#include "mlek/use_case/asr_zh/CtcDecoder.hpp"
#include "mlek/use_case/asr_zh/KaldiFbank.hpp"
#include "mlek/use_case/asr_zh/StreamingState.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

using namespace arm::app::asr_zh;

static void Require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void TestFbank()
{
    KaldiFbank extractor;
    std::array<float, 77 * 80> input{};
    std::array<float, 13 * 80> overlap{};
    constexpr size_t kSamples = sizeof(asr_zh_fixture::kPcm) / sizeof(int16_t);
    Require(!extractor.Begin(nullptr, 0), "Invalid PCM accepted");
    Require(extractor.Begin(asr_zh_fixture::kPcm, kSamples), "Fbank begin");
    Require(extractor.Frames() == 132, "snip_edges=False frame count");
    size_t chunks  = 0;
    float maxError = 0.f;
    while (extractor.Next(input.data(), input.size())) {
        if (chunks) {
            Require(std::equal(overlap.begin(), overlap.end(), input.begin()),
                    "Overlapping features were not reused exactly");
        }
        std::copy_n(input.begin() + 64 * 80, overlap.size(), overlap.begin());
        for (size_t row = 0; row < 77; ++row) {
            const size_t frame = chunks * 64 + row;
            if (frame >= extractor.Frames()) {
                Require(std::all_of(input.begin() + row * 80,
                                    input.begin() + (row + 1) * 80,
                                    [](float value) { return value == KaldiFbank::kPadding; }),
                        "Tail padding");
            }
            for (size_t golden = 0; golden < 7; ++golden) {
                if (asr_zh_fixture::kFrames[golden] == frame) {
                    for (size_t bin = 0; bin < 80; ++bin) {
                        maxError = std::max(maxError,
                                            std::abs(input[row * 80 + bin] -
                                                     asr_zh_fixture::kExpected[golden][bin]));
                    }
                }
            }
        }
        ++chunks;
    }
    std::printf("Fbank golden maximum absolute error: %.9f\n", maxError);
    Require(maxError < 0.003f, "Kaldi Fbank mismatch");
    Require(chunks == 3 && extractor.ComputedFrames() == extractor.Frames(),
            "Frames recomputed or skipped");
    Require(extractor.Begin(asr_zh_fixture::kPcm, 120), "Short reflected frame");
    Require(extractor.Next(input.data(), input.size()), "Short clip features");
    Require(std::all_of(input.begin(), input.end(), [](float x) { return std::isfinite(x); }),
            "Non-finite short clip");
    std::array<int16_t, 400> silence{};
    Require(extractor.Begin(silence.data(), silence.size()), "Silence");
    Require(extractor.Next(input.data(), input.size()), "Silence feature");
    Require(std::abs(input[0] - std::log(std::numeric_limits<float>::epsilon())) < 1e-5f,
            "Log floor");
}

static void TestDecoder()
{
    CtcDecoder decoder{assets::kTokens, 1000};
    for (const auto& chunk : assets::kExpectedPath) {
        for (auto token : chunk) {
            Require(decoder.AppendToken(token), "Actual PT2 token rejected");
        }
    }
    Require(decoder.Text() == assets::kExpectedText, "Byte-BPE reference mismatch");
    decoder.Reset();
    Require(decoder.Text().empty() && decoder.Path().empty(), "CTC reset");
    const unsigned char a[]       = {'a'};
    const unsigned char prefix[]  = {0xe4, 0xbd};
    const unsigned char suffix[]  = {0xa0};
    const unsigned char invalid[] = {0xff};
    const TokenBytes tokens[]     = {{nullptr, 0}, {a, 1}, {prefix, 2}, {suffix, 1}, {invalid, 1}};
    CtcDecoder tiny{tokens, 5};
    Require(tiny.AppendToken(1) && tiny.AppendToken(1), "Repeats");
    Require(tiny.Text() == "a", "Cross-call duplicate");
    Require(tiny.AppendToken(0) && tiny.AppendToken(1), "Blank separator");
    Require(tiny.Text() == "aa", "Blank must reset previous token");
    tiny.Reset();
    Require(tiny.AppendToken(2) && tiny.Text().empty(), "Incomplete UTF-8 handling");
    Require(tiny.AppendToken(3) && tiny.Text() == "\xe4\xbd\xa0", "UTF-8 across chunks");
    Require(tiny.AppendToken(4) && tiny.AppendToken(1), "Invalid byte test");
    Require(tiny.Text() == "\xe4\xbd\xa0"
                           "a",
            "UTF-8 recovery");
    Require(!tiny.AppendToken(5), "Out-of-vocabulary ID");
    std::array<float, 5> bad{};
    bad[0] = std::numeric_limits<float>::quiet_NaN();
    Require(!tiny.Append(bad.data(), 1, bad.size()), "NaN accepted");
}

static void TestState()
{
    StreamingState state;
    Require(!state.Reset() && !state.Advance(), "Unbound states accepted");
    size_t total = 0;
    for (size_t slot = 1; slot <= 74; ++slot) {
        total += StateBytes(slot);
    }
    Require(total == 2479620, "State contract changed");
    std::vector<unsigned char> input(total, 0xff);
    std::vector<unsigned char> output(total, 0x31);
    size_t offset = 0;
    for (size_t slot = 1; slot <= 74; ++slot) {
        const size_t bytes = StateBytes(slot);
        Require(!state.Bind(slot, input.data() + offset, bytes - 1, output.data() + offset, bytes),
                "Wrong byte size accepted");
        Require(state.Bind(slot, input.data() + offset, bytes, output.data() + offset, bytes),
                "State binding");
        offset += bytes;
    }
    Require(state.Ready() && state.Reset(), "State reset");
    Require(std::all_of(input.begin(), input.end(), [](unsigned char b) { return b == 0; }),
            "State not cleared");
    const int32_t count = 32;
    std::memcpy(output.data() + output.size() - 4, &count, 4);
    Require(state.Advance() && input == output, "State feedback changed bytes");
    Require(StateShape(73) == std::vector<size_t>{1, 128, 3, 19}, "Embedding state");
    Require(StateShape(74) == std::vector<size_t>{1}, "Counter state");
}

static void TestInt8State()
{
    StreamingState state;
    size_t total = 0;
    for (size_t slot = 1; slot <= 74; ++slot) {
        total += StateBytes(slot, true);
    }
    Require(total == 619908, "INT8 state contract changed");
    std::vector<unsigned char> input(total, 0xff);
    std::vector<unsigned char> output(total, 0x73);
    size_t offset = 0;
    for (size_t slot = 1; slot <= 74; ++slot) {
        const size_t bytes      = StateBytes(slot, true);
        const int32_t zeroPoint = slot == 74 ? 0 : static_cast<int32_t>(slot * 13 % 256) - 128;
        Require(!state.Bind(
                    slot, input.data() + offset, bytes, output.data() + offset, bytes, true, 128),
                "Invalid zero point accepted");
        Require(
            state.Bind(
                slot, input.data() + offset, bytes, output.data() + offset, bytes, true, zeroPoint),
            "INT8 state binding");
        offset += bytes;
    }
    Require(state.Reset(), "INT8 reset");
    offset = 0;
    for (size_t slot = 1; slot <= 74; ++slot) {
        const size_t bytes  = StateBytes(slot, true);
        const int zeroPoint = slot == 74 ? 0 : static_cast<int>(slot * 13 % 256) - 128;
        Require(std::all_of(input.begin() + offset,
                            input.begin() + offset + bytes,
                            [zeroPoint](unsigned char value) {
                                return value == static_cast<unsigned char>(zeroPoint);
                            }),
                "INT8 real-zero encoding mismatch");
        offset += bytes;
    }
    const int32_t count = 352;
    std::memcpy(output.data() + total - 4, &count, 4);
    Require(state.Advance() && input == output, "INT8 state feedback changed bytes");
}

int main()
{
    try {
        TestFbank();
        TestDecoder();
        TestState();
        TestInt8State();
        std::puts("ASR_ZH native processing tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
