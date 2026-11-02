/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#if defined(KWS_ZH_INPUT_UART)
#    include "../../../file_demo/include/UartFileInput.hpp"
#    include "BufAttributes.hpp"
#    include "hal.h"
#    include "mlek/common/ImageUtils.hpp"
#    include "mlek/fwk/executorch/EtModel.hpp"
#    include "mlek/use_case/kws_zh/KaldiFbank.hpp"
#    include "mlek/use_case/kws_zh/WakeWordDecoder.hpp"
#    include <algorithm>
#    include <cstdio>
#    include <cstdlib>
#    include <cstring>
#    include <string>
#    include <vector>

namespace arm::app::kws_zh {
namespace {
    void ShowWave(const int16_t* audio, size_t samples, uint64_t eventSample, bool hasEvent)
    {
        static uint8_t pixels[300 * 75 * 3] IFM_BUF_ATTRIBUTE;
        std::fill_n(pixels, sizeof(pixels), 0);
        for (size_t x = 0; x < 300; ++x) {
            int peak         = 0;
            const size_t end = std::min(samples, (x + 1) * samples / 300);
            for (size_t i = x * samples / 300; i < end; ++i) {
                peak = std::max(peak, std::abs(static_cast<int>(audio[i])));
            }
            const size_t height = peak * 35 / 32768;
            for (size_t y = 37 - height; y <= 37 + height; ++y) {
                pixels[(y * 300 + x) * 3 + 1] = 255;
            }
        }
        if (hasEvent && eventSample < samples) {
            const size_t x = eventSample * 300 / samples;
            for (size_t y = 0; y < 75; ++y) {
                pixels[(y * 300 + x) * 3]     = 255;
                pixels[(y * 300 + x) * 3 + 1] = 255;
            }
        }
        hal_display_show_image(pixels, 300, 75, 3, 9, 30, 1);
    }
} // namespace

bool ProcessFile(fwk::et::EtModel& model, const std::vector<std::string>& labels)
{
    static int16_t audio[30 * 16000] IFM_BUF_ATTRIBUTE;
    static KaldiFbank fbank;
    hal_display_clear(COLOR_BLACK);
    hal_display_set_text_color(COLOR_WHITE);
    const char* title = "Chinese KWS | send WAV";
    hal_display_show_text(title, std::strlen(title), 9, 0, false);
    for (;;) {
        size_t bytes      = 0;
        const uint32_t id = file_demo::Receive(
            "KWD1", 16000, 0, reinterpret_cast<uint8_t*>(audio), sizeof(audio), true, &bytes);
        const size_t samples = bytes / sizeof(int16_t);
        WakeWordDecoder decoder{KWS_ZH_SCORE_THRESHOLD,
                                KWS_ZH_MINIMUM_FRAMES,
                                static_cast<uint64_t>(KWS_ZH_COOLDOWN_MS) * 16};
        hal_display_clear(COLOR_BLACK);
        hal_display_set_text_color(COLOR_WHITE);
        hal_display_show_text(title, std::strlen(title), 9, 0, false);
        ShowWave(audio, samples, 0, false);
        size_t count = 0;
        for (size_t start = 0; start < samples; start += KWS_ZH_WINDOW_STRIDE) {
            const size_t available = std::min(samples - start, KaldiFbank::kWindowSamples);
            const auto input       = model.GetInputTensor(0);
            if (!fbank.Compute(
                    audio + start, available, input->GetData<float>(), input->GetNumElements()) ||
                !model.RunInference()) {
                std::printf("FILE_ERROR %" PRIu32 " INFERENCE\n", id);
                return false;
            }
            const auto events = decoder.Decode(
                model.GetOutputTensor(0)->GetData<float>(), KaldiFbank::kFrames, start, samples);
            for (const auto& event : events) {
                ++count;
                char line[64];
                std::snprintf(line,
                              sizeof(line),
                              "%s @ %.2f s",
                              labels[event.m_keyword].c_str(),
                              event.m_sample / 16000.0);
                info("WAKE label=%s score=%.6f time=%.3fs\n",
                     labels[event.m_keyword].c_str(),
                     event.m_score,
                     event.m_sample / 16000.0);
                ShowWave(audio, samples, event.m_sample, true);
                hal_display_show_box(0, 115, 320, 100, COLOR_BLACK);
                hal_display_set_text_color(COLOR_YELLOW);
                hal_display_show_text("NI HAO! ^_^", 11, 9, 120, false);
                hal_display_show_text(line, std::strlen(line), 9, 165, false);
            }
            if (start + available >= samples) {
                break;
            }
        }
        if (!count) {
            const char* message = "No keyword detected";
            hal_display_show_text(message, std::strlen(message), 9, 120, false);
        }
        std::printf("FILE_RESULT %" PRIu32 " %zu\n", id, count);
    }
}
} // namespace arm::app::kws_zh
#endif
