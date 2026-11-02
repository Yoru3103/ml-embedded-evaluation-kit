/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#if defined(KWS_ZH_INPUT_VSI)
#    include "AudioWindow.hpp"
#    include "BufAttributes.hpp"
#    include "FvpAudioInput.hpp"
#    include "GapRecovery.hpp"
#    include "UseCaseCommonUtils.hpp"
#    include "hal.h"
#    include "mlek/common/ImageUtils.hpp"
#    include "mlek/fwk/executorch/EtModel.hpp"
#    include "mlek/use_case/kws_zh/KaldiFbank.hpp"
#    include "mlek/use_case/kws_zh/WakeWordDecoder.hpp"
#    include <algorithm>
#    include <cmath>
#    include <cstdio>
#    include <cstring>
#    include <string>
#    include <vector>

namespace arm::app::kws_zh {
namespace {
    using Window = fvp_audio::AudioWindow<KaldiFbank::kWindowSamples, fvp_audio::kBlockSamples>;
    Window s_window;
    KaldiFbank s_streamFbank;
    static uint8_t s_wave[300 * 75 * 3] IFM_BUF_ATTRIBUTE;
    const char* s_status = nullptr;

    void Text(const char* text, uint32_t y, uint16_t color = COLOR_WHITE)
    {
        hal_display_set_text_color(color);
        hal_display_show_text(text, std::strlen(text), 9, y, false);
    }

    void ShowWave(const Window& window, uint64_t eventSample, bool hasEvent, uint32_t backlog)
    {
        std::fill_n(s_wave, sizeof(s_wave), 0);
        const size_t count = std::min(window.Count(), KaldiFbank::kWindowSamples);
        for (size_t x = 0; x < 300 && count > 0; ++x) {
            const size_t first = x * count / 300;
            const size_t last  = std::max(first + 1, (x + 1) * count / 300);
            int peak           = 0;
            for (size_t i = first; i < std::min(last, count); ++i) {
                peak = std::max(peak, std::abs(static_cast<int>(window.Data()[i])));
            }
            const uint32_t height = static_cast<uint32_t>(peak * 35 / 32768);
            for (uint32_t y = 37 - height; y <= 37 + height; ++y) {
                s_wave[(y * 300 + x) * 3 + 1] = 255;
            }
        }
        if (hasEvent && eventSample >= window.Start() && eventSample - window.Start() < count) {
            const uint32_t x = (eventSample - window.Start()) * 300 / count;
            for (uint32_t y = 0; y < 75; ++y) {
                s_wave[(y * 300 + x) * 3]     = 255;
                s_wave[(y * 300 + x) * 3 + 1] = 255;
            }
        }
        /* Publish one contiguous waveform instead of flushing 300 narrow columns. */
        hal_display_show_image(s_wave, 300, 75, 3, 9, 38, 1);
        hal_display_show_box(0, 180, 320, 30, COLOR_BLACK);
        char line[64];
        std::snprintf(line,
                      sizeof(line),
                      "Audio: %.2f s queue: %.0f ms",
                      static_cast<double>(window.Start() + window.Count()) / 16000,
                      static_cast<double>(backlog) / 16);
        Text(line, 192);
    }

    void ShowStatus(const char* status)
    {
        if (s_status && std::strcmp(s_status, status) == 0) {
            return;
        }
        s_status = status;
        hal_display_show_box(0, 15, 320, 15, COLOR_BLACK);
        Text(status, 15);
    }

    void ShowGreeting(const char* eventText)
    {
        /* Render only on a new detection. Waveform and gap updates leave this intact. */
        hal_display_show_box(0, 120, 320, 60, COLOR_BLACK);
        Text("NI HAO! ^_^", 120, COLOR_YELLOW);
        Text(eventText, 165, COLOR_YELLOW);
    }
} // namespace

bool ProcessVsi(fwk::et::EtModel& model, const std::vector<std::string>& labels)
{
    constexpr size_t hopSamples = KWS_ZH_STREAM_HOP_SAMPLES;
    static_assert(hopSamples >= 160 && hopSamples <= KaldiFbank::kWindowSamples);
    static_assert(hopSamples % 160 == 0);
    WakeWordDecoder decoder{KWS_ZH_SCORE_THRESHOLD,
                            KWS_ZH_MINIMUM_FRAMES,
                            static_cast<uint64_t>(KWS_ZH_COOLDOWN_MS) * 16};
    if (!fvp_audio::Open()) {
        printf_err("VSI audio bridge missing: configure mps4_board.v_path\n");
        return false;
    }
    fvp_audio::Block block;
    uint64_t expectedSample    = 0;
    uint64_t lastEventSample   = 0;
    uint64_t lastDisplaySample = 0;
    bool hasEvent              = false;
    bool started               = false;
    fvp_audio::GapRecovery recovery{KaldiFbank::kWindowSamples};
    char eventText[80] = "Listening - warming up...";
    s_window.Reset(0);
    hal_display_clear(COLOR_BLACK);
    s_status = nullptr;
    hal_display_show_box(0, 0, 320, 15, COLOR_BLACK);
    Text("Chinese KWS | live VSI", 0);
    Text("Timestamp = detector trigger", 215);
    ShowStatus("Listening - warming up...");
    Text("Say Nihao Wenwen", 150);
    ShowWave(s_window, 0, false, 0);
    info("KWS_VSI_READY sample_rate=16000 block=320 hop=%zu\n", hopSamples);
    while (fvp_audio::Read(block)) {
        const bool gap             = (block.flags & fvp_audio::DISCONTINUITY) ||
                                     (started && block.firstSample != expectedSample);
        const bool enteredRecovery = recovery.Observe(gap, block.samples);
        if (!started || gap) {
            if (enteredRecovery) {
                ShowStatus("Audio gap - warming up...");
            }
            decoder.Reset();
            s_window.Reset(block.firstSample);
            started = true;
        }
        expectedSample = block.firstSample + block.samples;
        if (!s_window.Append(block.pcm, block.samples)) {
            fvp_audio::Close();
            return false;
        }
        const bool eos = block.flags & fvp_audio::EOS;
        if (s_window.Ready() || (eos && s_window.Count() >= KaldiFbank::kFrameLength)) {
            const auto input   = model.GetInputTensor(0);
            const size_t valid = std::min(s_window.Count(), KaldiFbank::kWindowSamples);
            /* No profiler print/UART/LCD inside the inference interval. */
            if (!s_streamFbank.Compute(
                    s_window.Data(), valid, input->GetData<float>(), input->GetNumElements()) ||
                !model.RunInference()) {
                fvp_audio::Close();
                return false;
            }
            const float* scores = model.GetOutputTensor(0)->GetData<float>();
            if (!scores) {
                fvp_audio::Close();
                return false;
            }
            for (size_t i = 0; i < KaldiFbank::kFrames * labels.size(); ++i) {
                if (!std::isfinite(scores[i])) {
                    fvp_audio::Close();
                    return false;
                }
            }
            const auto events =
                decoder.Decode(scores, KaldiFbank::kFrames, s_window.Start(), expectedSample);
            ShowStatus("Listening...");
            for (const auto& event : events) {
                hasEvent        = true;
                lastEventSample = event.m_sample;
                std::snprintf(eventText,
                              sizeof(eventText),
                              "%s @ %.2f s",
                              labels[event.m_keyword].c_str(),
                              static_cast<double>(event.m_sample) / 16000);
                info("WAKE label=%s score=%.6f time=%.3fs queue_ms=%.1f\n",
                     labels[event.m_keyword].c_str(),
                     event.m_score,
                     static_cast<double>(event.m_sample) / 16000,
                     static_cast<double>(block.backlogSamples) / 16);
                info("NI HAO! ^_^\n");
                ShowGreeting(eventText);
            }
            ShowWave(s_window, lastEventSample, hasEvent, block.backlogSamples);
            lastDisplaySample = expectedSample;
            s_window.Advance(hopSamples);
        } else if (!recovery.Active() && expectedSample - lastDisplaySample >= 8000) {
            ShowWave(s_window, lastEventSample, hasEvent, block.backlogSamples);
            lastDisplaySample = expectedSample;
        }
        if (eos) {
            info("AUDIO_EOS sample=%" PRIu64 "\n", expectedSample);
            fvp_audio::Close();
            return true;
        }
    }
    fvp_audio::Close();
    printf_err("VSI audio input failed\n");
    return false;
}
} // namespace arm::app::kws_zh
#endif
