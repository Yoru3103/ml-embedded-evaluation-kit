/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "UseCaseHandler.hpp"
#include "BufAttributes.hpp"
#if defined(QRCODE_INPUT_SEMIHOST) || defined(QRCODE_INPUT_UART)
#    include "QrCodeImageRequest.hpp"
#endif

#include "DetectionDrawing.hpp"
#include "OutputDequantizationProfiler.hpp"
#include "QrCodePreProcessing.hpp"
#if defined(QRCODE_INPUT_UART)
#    include "../../../file_demo/include/UartFileInput.hpp"
#endif

#include "UseCaseCommonUtils.hpp"
#include "hal.h"
#include "mlek/common/ImageUtils.hpp"
#include "mlek/fwk/iface/Tensor.hpp"
#include "mlek/log/log_macros.h"
#include "mlek/use_case/yolov8_detection/YoloV8PostProcessing.hpp"
#include "sample_files.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace arm::app {
namespace {

    constexpr uint32_t classCount = 15;
#if defined(QRCODE_INPUT_SEMIHOST) || defined(QRCODE_INPUT_UART)
    static uint8_t s_runtimeImage[QRCODE_INPUT_WIDTH * QRCODE_INPUT_HEIGHT] IFM_BUF_ATTRIBUTE;

    void WaitForRequest()
    {
        /* Polling stays in the demo adapter, outside inference timing. */
        for (volatile uint32_t delay = 0; delay < 1000000U; ++delay) {}
    }
#endif

    bool CheckOutputShapes(fwk::iface::Model& model, size_t imageWidth, size_t imageHeight)
    {
        if (model.GetNumOutputs() != 4 || imageWidth % 32 != 0 || imageHeight % 32 != 0) {
            return false;
        }
        bool featureMaps[3]{};
        bool hasAngle     = false;
        size_t angleCount = 0;
        for (const size_t stride : {size_t{8}, size_t{16}, size_t{32}}) {
            angleCount += imageWidth / stride * (imageHeight / stride);
        }
        for (size_t index = 0; index < 4; ++index) {
            const auto tensor = model.GetOutputTensor(index);
            if (tensor == nullptr) {
                return false;
            }
            const auto shape = tensor->Shape();
            if (shape.size() == 3 && shape[0] == 1 &&
                ((shape[1] == 1 && shape[2] == angleCount) ||
                 (shape[1] == angleCount && shape[2] == 1))) {
                if (hasAngle) {
                    return false;
                }
                hasAngle = true;
                continue;
            }
            if (shape.size() != 4 || shape[0] != 1) {
                return false;
            }
            bool found = false;
            for (size_t map = 0; map < 3; ++map) {
                const size_t stride = size_t{8} << map;
                const size_t rows   = imageHeight / stride;
                const size_t cols   = imageWidth / stride;
                if (!featureMaps[map] &&
                    ((shape[1] == 47 && shape[2] == rows && shape[3] == cols) ||
                     (shape[1] == rows && shape[2] == cols && shape[3] == 47))) {
                    featureMaps[map] = true;
                    found            = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
        }
        return hasAngle && featureMaps[0] && featureMaps[1] && featureMaps[2];
    }

    const char* GetLabel(const std::vector<std::string>& labels, int classId)
    {
        if (classId < 0 || static_cast<size_t>(classId) >= labels.size()) {
            return "unknown";
        }
        return labels[static_cast<size_t>(classId)].c_str();
    }

    void PrintResults(const std::vector<yolov8_detection::DetectionResult>& results,
                      const std::vector<std::string>& labels)
    {
        info("Detections: %zu\n", results.size());
        for (size_t index = 0; index < results.size(); ++index) {
            const auto& result = results[index];
            info("%zu: class=%d label=%s score=%f cx=%f cy=%f w=%f h=%f angle=%f\n",
                 index,
                 result.classId,
                 GetLabel(labels, result.classId),
                 result.score,
                 result.centerX,
                 result.centerY,
                 result.width,
                 result.height,
                 result.angleRadians);
        }
    }

    void PresentResults(const std::vector<yolov8_detection::DetectionResult>& results,
                        const std::vector<std::string>& labels,
                        uint32_t x,
                        uint32_t y)
    {
        char line[32]{};
        hal_display_set_text_color(COLOR_GREEN);
        const int countLength =
            std::snprintf(line, sizeof(line), "Detections: %zu", results.size());
        if (countLength > 0) {
            hal_display_show_text(
                line, std::min(static_cast<size_t>(countLength), sizeof(line) - 1), x, y, false);
        }
        for (size_t index = 0; index < std::min(results.size(), size_t{5}); ++index) {
            const auto& result = results[index];
            const int length   = std::snprintf(line,
                                               sizeof(line),
                                               "%zu:%s %.2f",
                                               index,
                                               GetLabel(labels, result.classId),
                                               static_cast<double>(result.score));
            if (length > 0) {
                hal_display_show_text(line,
                                      std::min(static_cast<size_t>(length), sizeof(line) - 1),
                                      x,
                                      y + 16 + static_cast<uint32_t>(index) * 32,
                                      false);
                const int angleLength = std::snprintf(line,
                                                      sizeof(line),
                                                      "angle: %.1f deg",
                                                      static_cast<double>(result.angleRadians) *
                                                          180.0 / 3.14159265358979323846);
                hal_display_show_text(line,
                                      std::min(static_cast<size_t>(angleLength), sizeof(line) - 1),
                                      x,
                                      y + 32 + static_cast<uint32_t>(index) * 32,
                                      false);
            }
        }
    }

} /* anonymous namespace */

bool QrCodeHandler(ApplicationContext& ctx)
{
    auto& model          = ctx.Get<fwk::iface::Model&>("model");
    auto& profiler       = ctx.Get<Profiler&>("profiler");
    auto& outputProfiler = ctx.Get<OutputDequantizationProfiler&>("outputProfiler");
    const auto& labels   = ctx.Get<const std::vector<std::string>&>("labels");
    if (!model.IsInited() || model.GetNumInputs() != 1 || labels.size() != classCount) {
        printf_err("QR-code model requires one input and 15 labels\n");
        return false;
    }

    const auto input = model.GetInputTensor(0);
#if defined(QRCODE_INPUT_SEMIHOST) || defined(QRCODE_INPUT_UART)
    const uint32_t imageWidth  = QRCODE_INPUT_WIDTH;
    const uint32_t imageHeight = QRCODE_INPUT_HEIGHT;
#else
    const uint32_t imageWidth  = get_sample_img_width();
    const uint32_t imageHeight = get_sample_img_height();
#endif
    const size_t imageBytes = static_cast<size_t>(imageWidth) * imageHeight;
    if (input == nullptr || imageWidth == 0 || imageHeight == 0 ||
        input->Shape() != std::vector<size_t>{1, imageHeight, imageWidth, 1}) {
        printf_err("QR-code sample dimensions must match the grayscale NHWC model input\n");
        return false;
    }
    if (!CheckOutputShapes(model, imageWidth, imageHeight)) {
        printf_err("QR-code model requires three 47-channel heads and matching angle logits\n");
        return false;
    }

    std::vector<std::shared_ptr<fwk::iface::TensorIface>> outputs;
    for (size_t index = 0; index < 4; ++index) {
        outputs.push_back(model.GetOutputTensor(index));
    }
    std::vector<yolov8_detection::DetectionResult> results;
    const YoloV8PostProcessParams params{imageWidth,
                                         imageHeight,
                                         classCount,
                                         QRCODE_MAX_DETECTIONS,
                                         QRCODE_SCORE_THRESHOLD,
                                         QRCODE_NMS_THRESHOLD,
                                         true,  /* Raw angle logits, not objectness or radians. */
                                         8,     /* Four eight-bin DFL distance distributions. */
                                         true}; /* Oriented decoding and rotated NMS. */
    const bool modelDequantizes = outputs[0]->Type() == fwk::iface::TensorType::FP32;
    info("Output dequantization: %s\n", modelDequantizes ? "model operators" : "fused into Decode");
    YoloV8NewPostProcess postProcess{outputs, results, params};
    QrCodePreProcess preProcess{input};

    constexpr uint32_t imageX       = 10;
    constexpr uint32_t imageY       = 35;
    constexpr uint32_t screenWidth  = 320;
    constexpr uint32_t screenHeight = 240;
    constexpr uint32_t downscale    = QRCODE_DISPLAY_DOWNSCALE;
    if (downscale == 0 || imageWidth / downscale == 0 || imageHeight / downscale == 0 ||
        imageX + imageWidth / downscale + 10 >= screenWidth ||
        imageY + imageHeight / downscale > screenHeight) {
        printf_err("QR-code display layout does not fit the LCD\n");
        return false;
    }
    const uint32_t textX     = imageX + imageWidth / downscale + 10;
    const uint32_t textCellX = textX / 9 * 9;
    const uint32_t textCellY = 40 / 15 * 15;

#if defined(QRCODE_INPUT_SEMIHOST) || defined(QRCODE_INPUT_UART)
#    if !defined(QRCODE_INPUT_UART)
    uint32_t previous = 0;
    bool havePrevious = false;
#    endif
    info("QRCODE_READY width=%" PRIu32 " height=%" PRIu32 "\n", imageWidth, imageHeight);
    for (;;) {
#    if defined(QRCODE_INPUT_UART)
        uint32_t request = file_demo::Receive(
            "QRD1", imageWidth, imageHeight, s_runtimeImage, sizeof(s_runtimeImage));
#    else
        uint32_t request = 0;
        for (;;) {
            FILE* file = std::fopen(qrcode_demo::kInputFile, "rb");
            if (file) {
                const auto status = qrcode_demo::ReadRequest(file,
                                                             imageWidth,
                                                             imageHeight,
                                                             s_runtimeImage,
                                                             sizeof(s_runtimeImage),
                                                             havePrevious,
                                                             previous,
                                                             request);
                std::fclose(file);
                if (status == qrcode_demo::ReadStatus::ERROR) {
                    printf_err("QRCODE_ERROR INPUT (length, dimensions or CRC)\n");
                    return false;
                }
                if (status == qrcode_demo::ReadStatus::READY) {
                    break;
                }
            }
            WaitForRequest();
        }
#    endif
#    if !defined(QRCODE_INPUT_UART)
        previous     = request;
        havePrevious = true;
#    endif
        const uint8_t* image = s_runtimeImage;
        info("QRCODE_REQUEST id=%" PRIu32 "\n", request);
#else
    if (get_sample_img_total_bytes() != imageBytes) {
        printf_err("QR-code static sample size mismatch\n");
        return false;
    }
    for (uint32_t index = 0; index < get_sample_n_elements(); ++index) {
#    ifdef INTERACTIVE_MODE
        AwaitUserInput();
#    endif
        const uint8_t* image = get_sample_data_ptr(index);
        if (image == nullptr) {
            return false;
        }
        info("Running image %" PRIu32 ": %s\n", index, get_sample_data_filename(index));
#endif
        hal_display_clear(COLOR_BLACK);
#if defined(QRCODE_INPUT_SEMIHOST) || defined(QRCODE_INPUT_UART)
        const char* title = "QR / barcode detector";
        hal_display_set_text_color(COLOR_WHITE);
        hal_display_show_text(title, std::strlen(title), 9, 0, false);
#endif
        const std::string message{"Running inference"};
        hal_display_show_text(message.c_str(), message.size(), textX, 40, false);
        hal_display_show_image(image, imageWidth, imageHeight, 1, imageX, imageY, downscale);

        profiler.StartProfiling("PreProcessing", false);
        if (!preProcess.DoPreProcess(image, imageBytes)) {
            profiler.StopProfiling();
            printf_err("QR-code grayscale input preparation failed\n");
            return false;
        }
        profiler.StopProfiling();
        pmu_counters phases[4]{};
        outputProfiler.Reset();
        profiler.StartProfiling("InferenceAndPostProcessing", false);
        hal_pmu_get_counters(&phases[0]);
        if (!model.RunInference()) {
            profiler.StopProfiling();
            printf_err("QR-code inference failed\n");
            return false;
        }
        hal_pmu_get_counters(&phases[1]);
        if (!postProcess.DecodeDetections()) {
            profiler.StopProfiling();
            printf_err("QR-code post-processing failed\n");
            return false;
        }
        hal_pmu_get_cpu_counters(&phases[2]);
        postProcess.SuppressDetections();
        hal_pmu_get_cpu_counters(&phases[3]);
        profiler.StopProfiling();
        profiler.RecordProfiling(phases[0], phases[1], "Inference");
#if defined(CPU_PROFILE_ENABLED)
        if (modelDequantizes && !outputProfiler.Record(profiler)) {
            printf_err("Expected four output DEQUANTIZE timing events\n");
            return false;
        }
#endif
        profiler.RecordProfiling(phases[1], phases[2], "Decode", false);
        profiler.RecordProfiling(phases[2], phases[3], "NMS", false);

        PrintResults(results, labels);
        hal_display_show_box(textCellX, textCellY, screenWidth - textCellX, 15, COLOR_BLACK);
        PresentResults(results, labels, textX, 40);
        DrawDetectionBoxes(results, imageX, imageY, imageWidth, imageHeight, downscale);
        profiler.PrintProfilingResult();
#if defined(QRCODE_INPUT_SEMIHOST) || defined(QRCODE_INPUT_UART)
        info("QRCODE_RESULT id=%" PRIu32 " count=%zu\n", request, results.size());
#    if defined(QRCODE_INPUT_UART)
        std::printf("FILE_RESULT %" PRIu32 " %zu\n", request, results.size());
#    endif
#endif
    }
    return true;
}

} /* namespace arm::app */
