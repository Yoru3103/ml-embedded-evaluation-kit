/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ConformerVsiVad.hpp"
#include "Fake.hpp"
#include "FvpAudioInput.hpp"
#include <deque>
namespace arm::app::fvp_audio {
std::deque<Block> blocks;
unsigned closes = 0;
bool Open()
{ return true; }
bool Read(Block& block)
{
    if (blocks.empty())
        return false;
    block = blocks.front();
    blocks.pop_front();
    return true;
}
void Close()
{ ++closes; }
} // namespace arm::app::fvp_audio
#include "UseCaseHandler.hpp"
using namespace arm::app;
void Add(uint64_t first, int16_t value, uint32_t flags = 0, uint32_t count = 320)
{
    fvp_audio::Block b;
    b.firstSample = first;
    b.samples     = count;
    b.flags       = flags;
    std::fill(std::begin(b.pcm), std::end(b.pcm), value);
    fvp_audio::blocks.push_back(b);
}
void Reset()
{
    fvp_audio::blocks.clear();
    g_inferences = 0;
    g_captured.clear();
    g_lcd.clear();
    g_displayWrites = 0;
}
bool Run()
{
    ApplicationContext ctx;
    const auto before = fvp_audio::closes;
    const bool result = ClassifyVsiAudioHandler(ctx);
    assert(fvp_audio::closes == before + 1);
    return result;
}

uint64_t Speech(uint64_t first, int16_t value = 1234, uint32_t flags = 0)
{
    for (size_t i = 0; i < 15; ++i)
        Add(first + i * 320, value, i == 0 ? flags : 0);
    return first + 4800;
}

uint64_t Silence(uint64_t first)
{
    constexpr size_t blocks = arm::app::conformer_vsi::kPostRollSamples / 320;
    for (size_t i = 0; i < blocks; ++i)
        Add(first + i * 320, 0);
    return first + blocks * 320;
}

int main()
{
    // Clicks and short bursts are rejected, even when EOS requests a flush.
    Reset();
    Add(0, 1234);
    Add(320, 0, fvp_audio::EOS, 0);
    assert(Run());
    assert(g_inferences == 0);
    Reset();
    for (uint64_t i = 0; i < 7; ++i)
        Add(i * 320, 1234, i == 6 ? fvp_audio::EOS : 0);
    assert(Run());
    assert(g_inferences == 0);
    // A qualified sentence flushes once and retains its LCD result at EOS.
    Reset();
    const auto end = Speech(1000);
    Add(end, 0, fvp_audio::EOS, 0);
    assert(Run());
    assert(g_inferences == 1);
    assert(g_captured[0].size() == 4800);
    assert(g_captured[0].front() == 1234 && g_captured[0].back() == 1234);
    assert(std::find(g_lcd.begin(), g_lcd.end(), "decoded") != g_lcd.end());
    // EOS on a data block includes its PCM; silence-only EOS does not infer.
    Reset();
    Speech(0);
    fvp_audio::blocks.back().flags = fvp_audio::EOS;
    assert(Run());
    assert(g_inferences == 1);
    Reset();
    Add(0, 0);
    Add(320, 0, fvp_audio::EOS);
    assert(Run());
    assert(!g_inferences);
    // Explicit discontinuity with EOS aborts the old utterance, never flushes it.
    Reset();
    Speech(0);
    Add(4800, 0, fvp_audio::EOS | fvp_audio::DISCONTINUITY, 0);
    assert(Run());
    assert(!g_inferences);
    // Both explicit discontinuity and implicit gaps abort old PCM, then accept new speech.
    for (auto flag : {uint32_t{0}, fvp_audio::DISCONTINUITY}) {
        Reset();
        Speech(0);
        const auto next = Speech(9000, 2345, flag);
        Add(next, 0, fvp_audio::EOS, 0);
        assert(Run());
        assert(g_inferences == 1);
        assert(g_captured[0][0] == 2345 && g_captured[0][319] == 2345);
    }
    // Discontinuity also clears silence pre-roll from before the gap.
    Reset();
    Add(0, 99);
    Add(320, 99);
    const auto clean = Speech(9000, 2345, fvp_audio::DISCONTINUITY);
    Add(clean, 0, fvp_audio::EOS, 0);
    assert(Run());
    assert(g_captured[0].size() == 4800 && g_captured[0][0] == 2345);
    // ERROR wins over EOS, and a failed Read must abort rather than flush speech.
    Reset();
    Speech(0);
    Add(4800, 0, fvp_audio::ERROR | fvp_audio::EOS, 0);
    assert(!Run());
    assert(!g_inferences);
    Reset();
    Speech(0);
    assert(!Run());
    assert(!g_inferences);
    // Malformed empty non-EOS block is rejected, also without flushing speech.
    Reset();
    Speech(0);
    Add(4800, 0, 0, 0);
    assert(!Run());
    assert(!g_inferences);
    // Endpoint decodes before EOS; EOS does not duplicate the completed utterance.
    Reset();
    const auto stopped = Silence(Speech(0));
    Add(stopped + 9000, 0, fvp_audio::DISCONTINUITY);
    Add(stopped + 9320, 0, fvp_audio::EOS, 0);
    assert(Run());
    assert(g_inferences == 1);
    assert(std::find(g_lcd.begin(), g_lcd.end(), "decoded") != g_lcd.end());
    // A saturated live queue must not trigger LCD/UART work on every gap.
    // After thousands of broken blocks, the next complete phrase is still decoded.
    Reset();
    const auto firstResultEnd = Silence(Speech(0));
    for (uint64_t i = 0; i < 1500; ++i) {
        Add(firstResultEnd + i * 1600, 1234, fvp_audio::DISCONTINUITY);
    }
    const auto nextPhrase   = firstResultEnd + 1500 * 1600;
    const auto recoveredEnd = Silence(Speech(nextPhrase, 2345, fvp_audio::DISCONTINUITY));
    Add(recoveredEnd, 0, fvp_audio::EOS, 0);
    assert(Run());
    assert(g_inferences == 2);
    assert(g_captured[1][0] == 2345);
    assert(g_displayWrites < 20);
    std::puts("Native ASR handler EOS/discontinuity/error/LCD/recovery tests passed");
}
