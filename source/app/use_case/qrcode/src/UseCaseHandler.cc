/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "UseCaseHandler.hpp"

#include "DetectionDrawing.hpp"
#include "OutputDequantizationProfiler.hpp"
#include "QrCodePreProcessing.hpp"
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
#include <memory>
#include <string>
#include <vector>

namespace arm::app {
namespace {

    constexpr uint32_t imageSize  = 256;
    constexpr uint32_t classCount = 15;
    constexpr uint32_t angleCount = 32 * 32 + 16 * 16 + 8 * 8;

    bool CheckOutputShapes(fwk::iface::Model& model)
    {
        if (model.GetNumOutputs() != 4) {
            return false;
        }
        bool featureMaps[3]{};
        bool hasAngle               = false;
        constexpr size_t mapSizes[] = {32, 16, 8};
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
                const size_t side = mapSizes[map];
                if (!featureMaps[map] &&
                    ((shape[1] == 47 && shape[2] == side && shape[3] == side) ||
                     (shape[1] == side && shape[2] == side && shape[3] == 47))) {
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
        for (size_t index = 0; index < std::min(results.size(), size_t{9}); ++index) {
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
                                      y + static_cast<uint32_t>(index + 1) * 16,
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
    if (!model.IsInited() || model.GetNumInputs() != 1 || !CheckOutputShapes(model) ||
        labels.size() != classCount || get_sample_img_width() != imageSize ||
        get_sample_img_height() != imageSize ||
        get_sample_img_total_bytes() != imageSize * imageSize) {
        printf_err(
            "QR-code model/sample must be grayscale NHWC 256x256 with 47/47/47/angle outputs\n");
        return false;
    }

    const auto input = model.GetInputTensor(0);
    if (input == nullptr || input->Shape() != std::vector<size_t>{1, imageSize, imageSize, 1}) {
        printf_err("QR-code input must have NHWC shape [1, 256, 256, 1]\n");
        return false;
    }

    std::vector<std::shared_ptr<fwk::iface::TensorIface>> outputs;
    for (size_t index = 0; index < 4; ++index) {
        outputs.push_back(model.GetOutputTensor(index));
    }
    std::vector<yolov8_detection::DetectionResult> results;
    const YoloV8PostProcessParams params{imageSize,
                                         imageSize,
                                         classCount,
                                         QRCODE_MAX_DETECTIONS,
                                         QRCODE_SCORE_THRESHOLD,
                                         QRCODE_NMS_THRESHOLD};
    const bool modelDequantizes = outputs[0]->Type() == fwk::iface::TensorType::FP32;
    info("Output dequantization: %s\n", modelDequantizes ? "model operators" : "fused into Decode");
    YoloV8NewPostProcess postProcess{outputs, results, params};
    QrCodePreProcess preProcess{input};

    constexpr uint32_t imageX       = 10;
    constexpr uint32_t imageY       = 35;
    constexpr uint32_t screenWidth  = 320;
    constexpr uint32_t screenHeight = 240;
    constexpr uint32_t downscale    = QRCODE_DISPLAY_DOWNSCALE;
    if (downscale == 0 || imageSize / downscale == 0 ||
        imageX + imageSize / downscale + 10 >= screenWidth ||
        imageY + imageSize / downscale > screenHeight) {
        printf_err("QR-code display layout does not fit the LCD\n");
        return false;
    }
    const uint32_t textX     = imageX + imageSize / downscale + 10;
    const uint32_t textCellX = textX / 9 * 9;
    const uint32_t textCellY = 40 / 15 * 15;

    for (uint32_t index = 0; index < get_sample_n_elements(); ++index) {
#ifdef INTERACTIVE_MODE
        AwaitUserInput();
#endif
        const uint8_t* image = get_sample_data_ptr(index);
        if (image == nullptr) {
            return false;
        }
        info("Running image %" PRIu32 ": %s\n", index, get_sample_data_filename(index));
        hal_display_clear(COLOR_BLACK);
        const std::string message{"Running inference"};
        hal_display_show_text(message.c_str(), message.size(), textX, 40, false);
        hal_display_show_image(image, imageSize, imageSize, 1, imageX, imageY, downscale);

        profiler.StartProfiling("PreProcessing", false);
        if (!preProcess.DoPreProcess(image, get_sample_img_total_bytes())) {
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
        DrawDetectionBoxes(results, imageX, imageY, imageSize, imageSize, downscale);
        profiler.PrintProfilingResult();
    }
    return true;
}

} /* namespace arm::app */
