/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#if defined(ASR_ZH_INPUT_VSI) || defined(ASR_ZH_INPUT_UART)
#    include "BufAttributes.hpp"
#    if defined(ASR_ZH_INPUT_VSI)
#        include "../../asr/include/ConformerVsiVad.hpp"
#        include "FvpAudioInput.hpp"
#    else
#        include "../../../file_demo/include/UartFileInput.hpp"
#    endif
#    include "UseCaseHandler.hpp"
#    include "hal.h"
#    include "mlek/common/ImageUtils.hpp"
#    include "mlek/log/log_macros.h"
#    include <algorithm>
#    include <cstring>

namespace arm::app::asr_zh {
namespace {
    static int16_t s_pcm[30 * 16000] IFM_BUF_ATTRIBUTE;
    bool s_done   = false;
    bool s_failed = false;
#    if defined(ASR_ZH_INPUT_UART)
    uint32_t s_request = 0;
#    endif
#    if defined(ASR_ZH_INPUT_VSI)
    static conformer_vsi::Vad s_vad(s_pcm);
    fvp_audio::Block s_block{};
    size_t s_offset     = 0;
    uint64_t s_expected = 0;
    bool s_auto         = false;
    bool s_resetVad     = false;
#    endif
} // namespace

void ShowDemoStatus(const char* text)
{
    hal_display_clear(COLOR_BLACK);
    hal_display_set_text_color(COLOR_WHITE);
    const char* title = "Chinese ASR";
    // Clear the full row: hal_display_clear redraws the longer kit title.
    hal_display_show_box(0, 0, 320, 15, COLOR_BLACK);
    hal_display_show_text(title, std::strlen(title), 9, 0, false);
    hal_display_show_text(text, std::strlen(text), 9, 45, false);
}

bool OpenDemoInput()
{
    s_done   = false;
    s_failed = false;
    ShowDemoStatus("Send WAV / record speech");
#    if defined(ASR_ZH_INPUT_VSI)
    s_vad.Reset();
    s_block    = {};
    s_offset   = 0;
    s_expected = 0;
    s_auto     = false;
    s_resetVad = false;
    return fvp_audio::Open();
#    else
    return true;
#    endif
}

const int16_t* ReadDemoClip(uint32_t& count)
{
    count = 0;
    if (s_done) {
        return nullptr;
    }
#    if defined(ASR_ZH_INPUT_UART)
    size_t bytes = 0;
    s_request    = file_demo::Receive(
        "ZHD1", 16000, 0, reinterpret_cast<uint8_t*>(s_pcm), sizeof(s_pcm), true, &bytes);
    count = bytes / sizeof(int16_t);
    ShowDemoStatus("Processing - see terminal");
    return s_pcm;
#    else
    if (s_resetVad) {
        s_vad.Reset();
        s_resetVad = false;
    }
    for (;;) {
        if (s_offset == s_block.samples) {
            if (!fvp_audio::Read(s_block)) {
                printf_err("ASR_ZH_VSI_ERROR invalid audio input\n");
                s_failed = s_done = true;
                return nullptr;
            }
            s_offset = 0;
            if (s_block.flags & fvp_audio::WAITING) {
                // Advance simulated time between host polls so HDLCD can refresh.
                for (volatile uint32_t idle = 0; idle < 100000; ++idle) {}
                continue;
            }
            s_auto = (s_block.flags & fvp_audio::AUTO_UTTERANCES) != 0;
            const bool gap =
                (s_block.flags & fvp_audio::DISCONTINUITY) || s_block.firstSample != s_expected;
            s_expected = s_block.firstSample + s_block.samples;
            if (gap) {
                if (!s_auto) {
                    printf_err("ASR_ZH_VSI_ERROR input gap\n");
                    s_failed = s_done = true;
                    return nullptr;
                }
                // Discard the incomplete sentence once; rebuild pre-roll after the gap.
                s_vad.Reset();
            }
        }
        if (s_auto) {
            const bool speech =
                conformer_vsi::HasSpeechEnergy(s_block.pcm, s_block.samples, s_vad.Active());
            while (s_offset < s_block.samples) {
                const size_t index = s_offset++;
                if (s_vad.Push(s_block.pcm[index], speech, s_block.firstSample + index)) {
                    if (s_vad.Qualified()) {
                        count      = s_vad.Size();
                        s_resetVad = true;
                        ShowDemoStatus("Processing - see terminal");
                        return s_pcm;
                    }
                    s_vad.Reset();
                }
            }
        } else {
            if (s_block.samples > 30 * 16000 - count) {
                printf_err("ASR_ZH_VSI_ERROR over 30 seconds\n");
                s_failed = s_done = true;
                return nullptr;
            }
            std::copy_n(s_block.pcm, s_block.samples, s_pcm + count);
            count += s_block.samples;
            s_offset = s_block.samples;
        }
        if (s_block.flags & fvp_audio::EOS) {
            s_done = true;
            if (s_auto) {
                count = s_vad.Qualified() ? s_vad.Size() : 0;
            }
            if (count) {
                ShowDemoStatus("Processing - see terminal");
            }
            return count ? s_pcm : nullptr;
        }
    }
#    endif
}

void CompleteDemoClip(bool success)
{
    ShowDemoStatus(success ? "Complete - Chinese in terminal" : "Inference failed");
#    if defined(ASR_ZH_INPUT_UART)
    if (success) {
        printf("FILE_RESULT %" PRIu32 " 1\n", s_request);
    } else {
        printf("FILE_ERROR %" PRIu32 " INFERENCE\n", s_request);
    }
#    else
    if (!success) {
        printf("ASR_ZH_VSI_ERROR INFERENCE\n");
    }
#    endif
}

bool CloseDemoInput()
{
#    if defined(ASR_ZH_INPUT_VSI)
    fvp_audio::Close();
#    endif
    return !s_failed;
}
} // namespace arm::app::asr_zh
#endif
