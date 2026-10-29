/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#if defined(ASR_INPUT_UART)
#    include "AsrUartProtocol.hpp"
#    include "BufAttributes.hpp"
#    include "ConformerOpProfiler.hpp"
#    include "ConformerSlidingWindow.hpp"
#    include "UseCaseHandler.hpp"
#    include "hal.h"
#    if defined(ASR_UART_LCD)
#        include "mlek/common/ImageUtils.hpp"
#    endif
#    include "mlek/fwk/iface/Model.hpp"
#    include "mlek/use_case/asr/ConformerProcessing.hpp"
#    include "uart_config.h"
#    include <algorithm>
#    include <cinttypes>
#    include <cstring>

namespace arm::app {
namespace {
    /* This MPS4-only adapter deliberately stays outside HAL and the ML library.
     * UartGetc() maps CR to LF, so it cannot be used to receive binary PCM.
     * Read the same CMSDK DATA/STATE registers as the existing core-platform driver.
     */
    bool ReadRawByte(uint8_t& byte, uint64_t timeoutCycles)
    {
        auto* uart       = reinterpret_cast<volatile uint32_t*>(UART0_BASE);
        const auto start = timeoutCycles ? ReadConformerCpuCycles() : 0;
        while (!(uart[1] & 2U)) {
            if (timeoutCycles && ReadConformerCpuCycles() - start >= timeoutCycles) {
                return false;
            }
        }
        byte = static_cast<uint8_t>(uart[0]);
        return true;
    }

    /* Explicit DDR section: this must not consume the dedicated fast SRAM cache. */
    static int16_t s_audio[asr_serial::kMaxSamples] IFM_BUF_ATTRIBUTE;

    void PrintResult(uint32_t id, uint64_t npu, uint64_t total, const std::string& text)
    { printf("ASR_RESULT %" PRIu32 " %" PRIu64 " %" PRIu64 " %s\n", id, npu, total, text.c_str()); }

    void ShowUartStatus(const char* status)
    {
#    if defined(ASR_UART_LCD)
        hal_display_clear(COLOR_BLACK);
        hal_display_set_text_color(COLOR_WHITE);
        const char* title = "Conformer ASR | FVP";
        hal_display_show_text(title, std::strlen(title), 9, 0, false);
        hal_display_show_text(status, std::strlen(status), 9, 30, false);
#    else
        (void)status;
#    endif
    }

    void ShowUartResult(const std::string& text, uint64_t npu, uint64_t total)
    {
#    if defined(ASR_UART_LCD)
        constexpr size_t columns      = 33;
        constexpr size_t visibleLines = 8;
        std::vector<std::string> lines;
        for (size_t offset = 0; offset < text.size();) {
            while (offset < text.size() && text[offset] == ' ') {
                ++offset;
            }
            if (offset == text.size()) {
                break;
            }
            size_t end = std::min(offset + columns, text.size());
            if (end < text.size() && text[end] != ' ') {
                const auto boundary = text.rfind(' ', end);
                if (boundary != std::string::npos && boundary > offset) {
                    end = boundary;
                }
            }
            lines.emplace_back(text.substr(offset, end - offset));
            offset = end;
        }
        ShowUartStatus(lines.size() > visibleLines ? "Complete - last page (see UART)"
                                                   : "Complete");
        hal_display_set_text_color(COLOR_GREEN);
        const size_t first = lines.size() > visibleLines ? lines.size() - visibleLines : 0;
        for (size_t row = first; row < lines.size(); ++row) {
            hal_display_show_text(lines[row].c_str(),
                                  lines[row].size(),
                                  9,
                                  60 + static_cast<uint32_t>(row - first) * 15,
                                  false);
        }
        if (text.empty()) {
            const char* empty = "(no speech decoded)";
            hal_display_show_text(empty, std::strlen(empty), 9, 60, false);
        }
        hal_display_set_text_color(COLOR_WHITE);
        char timing[64];
        snprintf(timing,
                 sizeof(timing),
                 "NPU.Driver: %" PRIu64 ".%03" PRIu64 " ms",
                 npu / 1000,
                 npu % 1000);
        hal_display_show_text(timing, std::strlen(timing), 9, 195, false);
        snprintf(timing,
                 sizeof(timing),
                 "Processing: %" PRIu64 ".%03" PRIu64 " ms",
                 total / 1000,
                 total % 1000);
        hal_display_show_text(timing, std::strlen(timing), 9, 210, false);
        const char* note = "Simulated times (fast mode)";
        hal_display_show_text(note, std::strlen(note), 9, 225, false);
#    else
        (void)text;
        (void)npu;
        (void)total;
#    endif
    }
} // namespace

bool ClassifyUartAudioHandler(ApplicationContext& ctx)
{
    using namespace asr_serial;
    auto& model              = ctx.Get<fwk::iface::Model&>("model");
    auto& tracer             = ctx.Get<ConformerOpProfiler&>("opProfiler");
    auto& labels             = ctx.Get<const std::vector<std::string>&>("labels");
    const uint64_t frequency = get_mps4_core_clock();
#    if ASR_UART_FPGA
    const uint64_t receiveTimeoutCycles = 2 * frequency;
#    else
    /* Host-paced TCP I/O cannot use a deadline in simulated CPU time. An aborted
     * FVP upload must be recovered by restarting the simulation, not by retrying
     * a new header into a partially received binary payload.
     */
    const uint64_t receiveTimeoutCycles = 0;
#    endif
    const auto shape = model.GetInputTensor(0)->Shape();
    if (!frequency || shape != std::vector<size_t>{1, 700, 80} ||
        model.GetOutputTensor(0)->Shape() != std::vector<size_t>{1, 174, 129} ||
        labels.size() != 128) {
        printf_err("UART demo requires a 700x80 Conformer and a CPU profiling clock\n");
        return false;
    }
    ConformerPreProcess<int16_t> pre(
        model.GetInputTensor(0), model.GetInputTensor(1), 512, 160, 700);
    std::string text;
    ConformerPostProcess post(model.GetOutputTensor(0), model.GetOutputTensor(1), labels, text);
    hal_pmu_reset();
    hal_pmu_final();
    Request previous{};
    bool cached        = false;
    uint64_t cachedNpu = 0, cachedTotal = 0;
    std::string cachedText;
    ShowUartStatus("Ready - send a WAV");
    printf("ASR_UART_READY 1\n");
    for (;;) {
        HeaderParser parser;
        uint8_t byte = 0;
        while (true) {
            if (!ReadRawByte(byte, receiveTimeoutCycles)) {
                parser.Reset();
                continue;
            }
            if (parser.Empty() && byte == '?') {
                printf("ASR_HELLO 1 %" PRIu32 " %" PRIu32 " %zu\n",
                       kSampleRate,
                       kMaxSamples,
                       kBlockBytes);
                continue;
            }
            if (parser.Push(byte)) {
                break;
            }
        }
        const auto request = parser.Get();
        if (!request.Valid()) {
            printf("ASR_ERROR %" PRIu32 " LENGTH\n", request.id);
            continue;
        }
        if (cached && request.id == previous.id) {
            if (request.crc == previous.crc && request.samples == previous.samples) {
                PrintResult(request.id, cachedNpu, cachedTotal, cachedText);
                ShowUartResult(cachedText, cachedNpu, cachedTotal);
            } else {
                printf("ASR_ERROR %" PRIu32 " REQUEST_ID\n", request.id);
            }
            continue;
        }
        ShowUartStatus("Receiving audio...");
        printf("ASR_READY %" PRIu32 "\n", request.id);
        auto* bytes             = reinterpret_cast<uint8_t*>(s_audio);
        const size_t totalBytes = request.samples * sizeof(int16_t);
        uint32_t crc            = 0xFFFFFFFFU;
        size_t received         = 0;
        bool complete           = true;
        while (received < totalBytes) {
            const auto count = std::min(kBlockBytes, totalBytes - received);
            size_t i         = 0;
            for (; i < count; ++i) {
                if (!ReadRawByte(bytes[received + i], receiveTimeoutCycles)) {
                    complete = false;
                    break;
                }
            }
            if (!complete) {
                break;
            }
            crc = UpdateCrc(crc, bytes + received, count);
            received += count;
            printf("ASR_ACK %" PRIu32 " %zu\n", request.id, received);
        }
        if (!complete || (crc ^ 0xFFFFFFFFU) != request.crc) {
            printf("ASR_ERROR %" PRIu32 " %s\n", request.id, complete ? "CRC" : "TIMEOUT");
            ShowUartStatus("Upload error - see UART");
            continue;
        }
        ShowUartStatus("Processing...");
        /* No transfer, display or UART output inside this elapsed interval. */
        const auto start   = ReadConformerCpuCycles();
        bool success       = true;
        uint64_t npuCycles = 0;
        conformer_window::Decoder decoder(labels);
        const bool sliding = request.samples > conformer_window::kWindowSamples;
        for (size_t offset = 0;; offset += conformer_window::kHopSamples) {
            const auto window = conformer_window::GetWindow(request.samples, offset);
            success           = pre.DoPreProcess(s_audio + offset, window.samples);
            if (success) {
                tracer.Reset();
                success = model.RunInference();
                tracer.Stop();
                const auto& npu = tracer.Get(ConformerOpProfiler::Operation::ETHOS_U);
                success         = success && npu.calls == 1;
                npuCycles += npu.cycles;
            }
            if (success) {
                if (sliding) {
                    success =
                        decoder.Append(static_cast<float*>(model.GetOutputTensor(0)->GetData()),
                                       conformer_window::kOutputRows,
                                       labels.size() + 1,
                                       window.firstRow,
                                       window.endRow);
                } else {
                    /* Preserve the existing short-sentence decode, including padding. */
                    success = post.DoPostProcess();
                }
            }
            if (!success || window.last) {
                break;
            }
        }
        if (success && sliding) {
            text = decoder.Result();
        }
        const auto elapsed = ReadConformerCpuCycles() - start;
        if (!success) {
            printf("ASR_ERROR %" PRIu32 " INFERENCE\n", request.id);
            ShowUartStatus("Inference error - see UART");
            continue;
        }
        previous = request;
        cached   = true;
        cachedNpu =
            (npuCycles / frequency) * 1000000ULL + (npuCycles % frequency) * 1000000ULL / frequency;
        cachedTotal =
            (elapsed / frequency) * 1000000ULL + (elapsed % frequency) * 1000000ULL / frequency;
        cachedText = text;
        PrintResult(request.id, cachedNpu, cachedTotal, cachedText);
        /* Result transmission and LCD rendering are both outside processing timing.
         * Reply first to avoid delaying the host result for the LCD refresh.
         */
        ShowUartResult(cachedText, cachedNpu, cachedTotal);
    }
}
} // namespace arm::app
#endif
