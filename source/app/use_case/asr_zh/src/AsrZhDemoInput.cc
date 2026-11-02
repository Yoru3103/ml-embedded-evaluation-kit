/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#if defined(ASR_ZH_INPUT_VSI) || defined(ASR_ZH_INPUT_UART)
#    include "BufAttributes.hpp"
#    if defined(ASR_ZH_INPUT_VSI)
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
    bool s_done        = false;
    bool s_failed      = false;
    uint32_t s_request = 0;
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
    uint64_t expected = 0;
    for (;;) {
        fvp_audio::Block block{};
        if (!fvp_audio::Read(block) || (block.flags & fvp_audio::ERROR) ||
            (block.flags & fvp_audio::DISCONTINUITY) || block.firstSample != expected ||
            block.samples > fvp_audio::kBlockSamples || block.samples > 30 * 16000 - count) {
            printf_err("ASR_ZH_VSI_ERROR input gap, invalid audio or over 30 seconds\n");
            s_failed = true;
            s_done   = true;
            return nullptr;
        }
        std::copy_n(block.pcm, block.samples, s_pcm + count);
        count += block.samples;
        expected += block.samples;
        if (block.flags & fvp_audio::EOS) {
            s_done = true;
            ShowDemoStatus("Processing - see terminal");
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
