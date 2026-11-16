/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#if defined(ASR_INPUT_VSI)
#    include "BufAttributes.hpp"
#    include "ConformerSlidingWindow.hpp"
#    include "ConformerVsiVad.hpp"
#    include "FvpAudioInput.hpp"
#    include "GapRecovery.hpp"
#    include "UseCaseHandler.hpp"
#    include "hal.h"
#    include "mlek/common/ImageUtils.hpp"
#    include "mlek/fwk/iface/Model.hpp"
#    include "mlek/log/log_macros.h"
#    include "mlek/use_case/asr/ConformerProcessing.hpp"
#    if defined(ASR_ET_STAGE_PROFILING)
#        include "ConformerOpProfiler.hpp"
#    endif
#    include <algorithm>
#    include <cinttypes>
#    include <cstring>
#    include <limits>

namespace arm::app {
namespace {
    /* Keep the 30-second utterance in DDR, outside the NPU fast SRAM cache. */
    static int16_t s_audio[conformer_vsi::kMaxSamples] IFM_BUF_ATTRIBUTE;
    const char* s_status = nullptr;

    void ShowStatus(const char* status)
    {
        if (s_status && std::strcmp(s_status, status) == 0) {
            return;
        }
        s_status = status;
        /* Status transitions never erase the last successful transcript. */
        hal_display_show_box(0, 15, 320, 30, COLOR_BLACK);
        hal_display_set_text_color(COLOR_WHITE);
        hal_display_show_text(status, std::strlen(status), 9, 30, false);
    }

    void ShowResult(const std::string& text)
    {
        if (text.empty()) {
            ShowStatus("Listening - no new text");
            return;
        }
        hal_display_show_box(0, 60, 320, 180, COLOR_BLACK);
        ShowStatus("Listening - last result:");
        hal_display_set_text_color(COLOR_GREEN);
        size_t offset = 0;
        for (uint32_t row = 0; row < 10 && offset < text.size(); ++row) {
            while (offset < text.size() && text[offset] == ' ') {
                ++offset;
            }
            size_t end = std::min(offset + 33, text.size());
            if (end < text.size() && text[end] != ' ') {
                const auto boundary = text.rfind(' ', end);
                if (boundary != std::string::npos && boundary > offset) {
                    end = boundary;
                }
            }
            hal_display_show_text(text.c_str() + offset, end - offset, 9, 60 + row * 15, false);
            offset = end;
        }
        if (offset < text.size()) {
            hal_display_set_text_color(COLOR_WHITE);
            const char* notice = "More text in UART log";
            hal_display_show_text(notice, std::strlen(notice), 9, 225, false);
        }
    }

    struct InputSession {
        ~InputSession()
        { fvp_audio::Close(); }
    };
} // namespace

bool ClassifyVsiAudioHandler(ApplicationContext& ctx)
{
    using namespace conformer_vsi;
    auto& model  = ctx.Get<fwk::iface::Model&>("model");
    auto& labels = ctx.Get<const std::vector<std::string>&>("labels");
    if (!model.IsInited() || model.GetInputTensor(0)->Shape() != std::vector<size_t>{1, 700, 80} ||
        model.GetOutputTensor(0)->Shape() != std::vector<size_t>{1, 174, 129} ||
        labels.size() != 128 || ctx.Get<uint32_t>("melSpecWindowSize") != 512 ||
        ctx.Get<uint32_t>("melSpecHopSize") != 160 || ctx.Get<uint32_t>("chunkSize") != 700) {
        printf_err("VSI ASR requires the 16 kHz 700x80 Conformer configuration\n");
        return false;
    }
    ConformerPreProcess<int16_t> pre(
        model.GetInputTensor(0), model.GetInputTensor(1), 512, 160, 700);
    std::string text;
    ConformerPostProcess post(model.GetOutputTensor(0), model.GetOutputTensor(1), labels, text);
    // The enlarged history must not consume the Cortex-M's 32 KiB call stack.
    static Vad vad(s_audio);
    vad.Reset();
    if (!fvp_audio::Open()) {
        printf_err("ASR_VSI_ERROR OPEN\n");
        fvp_audio::Close();
        return false;
    }
    InputSession session;
    uint64_t utterance = 0;
    uint64_t expected  = 0;
    bool havePosition  = false;
    fvp_audio::GapRecovery recovery{16000};

    /* Decode before the next pull. Host acquisition/backpressure belongs to the bridge. */
    auto decode = [&](const char* reason, uint32_t backlog) -> bool {
        if (!vad.Active()) {
            return true;
        }
        if (!vad.Qualified()) {
            vad.Reset();
            ShowStatus("Listening...");
            return true;
        }
        const size_t captured = vad.Size();
        const size_t samples  = std::max(captured, kMinSamples);
        std::fill(s_audio + captured, s_audio + samples, 0);
        ++utterance;
        printf("ASR_VSI_UTTERANCE %" PRIu64 " %" PRIu64 " %" PRIu64 " %s backlog=%" PRIu32 "\n",
               utterance,
               vad.First(),
               vad.First() + captured,
               reason,
               backlog);
        ShowStatus("Processing sentence...");
        text.clear();
        conformer_window::Decoder decoder(labels);
        const bool sliding = samples > conformer_window::kWindowSamples;
        for (size_t offset = 0;; offset += conformer_window::kHopSamples) {
            const auto window = conformer_window::GetWindow(samples, offset);
            if (!pre.DoPreProcess(s_audio + offset, window.samples)) {
                return false;
            }
#    if defined(ASR_ET_STAGE_PROFILING)
            auto& tracer = ctx.Get<ConformerOpProfiler&>("opProfiler");
            tracer.Reset();
#    endif
            const bool success = model.RunInference();
#    if defined(ASR_ET_STAGE_PROFILING)
            tracer.Stop();
#    endif
            if (!success) {
                return false;
            }
            if (sliding) {
                if (!decoder.Append(static_cast<float*>(model.GetOutputTensor(0)->GetData()),
                                    conformer_window::kOutputRows,
                                    labels.size() + 1,
                                    window.firstRow,
                                    window.endRow)) {
                    return false;
                }
            } else if (!post.DoPostProcess()) {
                return false;
            }
            if (window.last) {
                break;
            }
        }
        if (sliding) {
            text = decoder.Result();
        }
        printf("ASR_VSI_RESULT %" PRIu64 " %" PRIu64 " %" PRIu64 " %s\n",
               utterance,
               vad.First(),
               vad.First() + captured,
               text.c_str());
        ShowResult(text);
        vad.Reset();
        return true;
    };

    hal_display_clear(COLOR_BLACK);
    s_status          = nullptr;
    const char* title = "Conformer ASR | VSI";
    hal_display_set_text_color(COLOR_WHITE);
    // Clear the full row: hal_display_clear redraws the longer kit title.
    hal_display_show_box(0, 0, 320, 15, COLOR_BLACK);
    hal_display_show_text(title, std::strlen(title), 9, 0, false);
    ShowStatus("Listening...");
    printf("ASR_VSI_READY 1 16000 320\n");
    for (;;) {
        fvp_audio::Block block{};
        if (!fvp_audio::Read(block) || (block.flags & fvp_audio::ERROR) ||
            block.samples > fvp_audio::kBlockSamples ||
            block.firstSample > std::numeric_limits<uint64_t>::max() - block.samples ||
            (!block.samples && !(block.flags & fvp_audio::EOS))) {
            printf_err("ASR_VSI_ERROR READ (utterance aborted)\n");
            ShowStatus("Audio input error");
            vad.Reset();
            return false;
        }
        const bool gap             = (block.flags & fvp_audio::DISCONTINUITY) ||
                                     (havePosition && block.firstSample != expected);
        const bool enteredRecovery = recovery.Observe(gap, block.samples);
        if (gap) {
            /* No UART in the overrun path: logging can itself prevent queue recovery. */
            vad.Reset();
            if (enteredRecovery) {
                ShowStatus("Listening - audio gap");
            }
        }
        expected          = block.firstSample + block.samples;
        havePosition      = true;
        const bool speech = HasSpeechEnergy(block.pcm, block.samples, vad.Active());
        for (size_t i = 0; i < block.samples; ++i) {
            const bool wasActive = vad.Active();
            if (vad.Push(block.pcm[i], speech, block.firstSample + i)) {
                const char* reason = vad.Size() == kMaxSamples ? "LIMIT" : "SILENCE";
                if (!decode(reason, block.backlogSamples)) {
                    printf_err("ASR_VSI_ERROR INFERENCE\n");
                    return false;
                }
            } else if (!wasActive && vad.Active()) {
                ShowStatus("Capturing speech...");
            }
        }
        if (block.flags & fvp_audio::EOS) {
            const bool success = decode("EOS", block.backlogSamples);
            printf("ASR_VSI_EOS %" PRIu64 " success=%u\n", expected, success ? 1U : 0U);
            return success;
        }
    }
}
} // namespace arm::app
#endif
