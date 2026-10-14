/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "UseCaseHandler.hpp"

#include "DetectionDrawing.hpp"
#include "UseCaseCommonUtils.hpp"
#include "hal.h"
#include "mlek/common/ImageUtils.hpp"
#include "mlek/log/log_macros.h"
#include "mlek/use_case/yolov8_detection/YoloV8PostProcessing.hpp"
#include "mlek/use_case/yolov8_detection/YoloV8PreProcessing.hpp"
#include "sample_files.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace arm::app {

namespace {

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

        for (size_t index = 0; index < results.size(); index++) {
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
                        uint32_t startX,
                        uint32_t startY)
    {
        constexpr uint32_t lineHeight        = 16;
        constexpr size_t maxDisplayedResults = 9;
        char line[32]{};

        hal_display_set_text_color(COLOR_GREEN);
        const int countLength =
            std::snprintf(line, sizeof(line), "Detections: %zu", results.size());
        if (countLength > 0) {
            hal_display_show_text(line,
                                  std::min(static_cast<size_t>(countLength), sizeof(line) - 1),
                                  startX,
                                  startY,
                                  false);
        }

        const size_t displayedResults = std::min(results.size(), maxDisplayedResults);
        for (size_t index = 0; index < displayedResults; ++index) {
            const auto& result   = results[index];
            const int lineLength = std::snprintf(line,
                                                 sizeof(line),
                                                 "%zu:%s %.2f",
                                                 index,
                                                 GetLabel(labels, result.classId),
                                                 static_cast<double>(result.score));
            if (lineLength > 0) {
                hal_display_show_text(line,
                                      std::min(static_cast<size_t>(lineLength), sizeof(line) - 1),
                                      startX,
                                      startY + static_cast<uint32_t>(index + 1) * lineHeight,
                                      false);
            }
        }
    }

    bool GetInputImageDimensions(const std::vector<size_t>& shape,
                                 uint32_t& imageWidth,
                                 uint32_t& imageHeight)
    {
        if (shape.size() != 4 || shape[0] != 1) {
            return false;
        }

        if ((shape[1] == 1 || shape[1] == 3) &&
            ((shape[3] != 1 && shape[3] != 3) || shape[2] == shape[3])) {
            imageHeight = static_cast<uint32_t>(shape[2]);
            imageWidth  = static_cast<uint32_t>(shape[3]);
            return true;
        }

        if ((shape[3] == 1 || shape[3] == 3) &&
            ((shape[1] != 1 && shape[1] != 3) || shape[1] == shape[2])) {
            imageHeight = static_cast<uint32_t>(shape[1]);
            imageWidth  = static_cast<uint32_t>(shape[2]);
            return true;
        }

        return false;
    }

} /* anonymous namespace */

bool YoloV8DetectionHandler(ApplicationContext& ctx)
{
    auto& model        = ctx.Get<fwk::iface::Model&>("model");
    auto& profiler     = ctx.Get<Profiler&>("profiler");
    const auto& labels = ctx.Get<const std::vector<std::string>&>("labels");

    if (!model.IsInited()) {
        printf_err("Model is not initialised\n");
        return false;
    }

    if (model.GetNumInputs() != 1) {
        printf_err("YOLOv8 model must have one input tensor\n");
        return false;
    }

    if (model.GetNumOutputs() != 1 && model.GetNumOutputs() != 3 && model.GetNumOutputs() != 4) {
        printf_err("YOLOv8 model must have one output or a three/four-output DFL head\n");
        return false;
    }

    auto inputTensor = model.GetInputTensor(0);
    if (inputTensor == nullptr) {
        printf_err("YOLOv8 input tensor is unavailable\n");
        return false;
    }

    const auto inputShape = inputTensor->Shape();
    uint32_t imageWidth   = 0;
    uint32_t imageHeight  = 0;

    if (!GetInputImageDimensions(inputShape, imageWidth, imageHeight)) {
        printf_err("Expected square RGB or grayscale NCHW/NHWC input\n");
        return false;
    }
    if (imageWidth == 0 || imageHeight == 0) {
        printf_err("YOLOv8 input dimensions must be non-zero\n");
        return false;
    }

    if (get_sample_img_width() != imageWidth || get_sample_img_height() != imageHeight) {
        printf_err("Generated image dimensions do not match model input\n");
        return false;
    }

    YoloV8PreProcess preProcess{inputTensor};

    if (labels.size() != YOLOV8_NUM_CLASSES) {
        printf_err("YOLOv8 label count does not match the configured class count\n");
        return false;
    }

    std::vector<yolov8_detection::DetectionResult> results;
    YoloV8PostProcessParams params{imageWidth,
                                   imageHeight,
                                   YOLOV8_NUM_CLASSES,
                                   YOLOV8_MAX_DETECTIONS,
                                   YOLOV8_SCORE_THRESHOLD,
                                   YOLOV8_NMS_THRESHOLD};
    std::unique_ptr<BasePostProcess> postProcess;
    if (model.GetNumOutputs() == 1) {
        postProcess =
            std::make_unique<YoloV8PostProcess>(model.GetOutputTensor(0), results, params);
    } else {
        std::vector<std::shared_ptr<fwk::iface::TensorIface>> outputTensors;
        outputTensors.reserve(model.GetNumOutputs());
        for (size_t outputIndex = 0; outputIndex < model.GetNumOutputs(); ++outputIndex) {
            outputTensors.push_back(model.GetOutputTensor(outputIndex));
        }
        postProcess = std::make_unique<YoloV8NewPostProcess>(outputTensors, results, params);
    }

    constexpr uint32_t imageStartX   = 10;
    constexpr uint32_t imageStartY   = 35;
    constexpr uint32_t textStartY    = 40;
    constexpr uint32_t displayWidth  = 320;
    constexpr uint32_t displayHeight = 240;
    constexpr uint32_t downscale     = YOLOV8_DISPLAY_DOWNSCALE;
    if (downscale == 0) {
        printf_err("YOLOv8 display downscale must be greater than zero\n");
        return false;
    }

    const uint32_t displayedImageWidth  = imageWidth / downscale;
    const uint32_t displayedImageHeight = imageHeight / downscale;
    const uint32_t textStartX           = imageStartX + displayedImageWidth + 10;
    if (textStartX >= displayWidth || imageStartY + displayedImageHeight > displayHeight) {
        printf_err("Displayed image does not fit the Corstone-320 HDLCD\n");
        return false;
    }
    /* HDLCD renders text at the origin of its 9x15 character cell. */
    constexpr uint32_t textCellWidth  = 9;
    constexpr uint32_t textCellHeight = 15;
    const uint32_t textPanelX         = textStartX / textCellWidth * textCellWidth;
    const uint32_t textPanelY         = textStartY / textCellHeight * textCellHeight;

    for (uint32_t index = 0; index < get_sample_n_elements(); ++index) {
#ifdef INTERACTIVE_MODE
        AwaitUserInput();
#endif

        const uint8_t* image = get_sample_data_ptr(index);
        if (image == nullptr) {
            printf_err("Failed to get input image\n");
            return false;
        }

        info("Running image %" PRIu32 ": %s\n", index, get_sample_data_filename(index));

        hal_display_clear(COLOR_BLACK);
        const std::string inferenceMessage{"Running inference"};
        hal_display_show_text(
            inferenceMessage.c_str(), inferenceMessage.size(), textStartX, textStartY, false);
        hal_display_show_image(
            image, imageWidth, imageHeight, 3, imageStartX, imageStartY, downscale);

        if (!preProcess.DoPreProcess(image, get_sample_img_total_bytes())) {
            printf_err("Pre-processing failed\n");
            return false;
        }

        profiler.StartProfiling("InferenceAndPostProcessing");
        if (!model.RunInference()) {
            profiler.StopProfiling();
            printf_err("Inference failed\n");
            return false;
        }

        if (!postProcess->DoPostProcess()) {
            profiler.StopProfiling();
            printf_err("Post-processing failed\n");
            return false;
        }
        profiler.StopProfiling();

        PrintResults(results, labels);
        hal_display_show_box(
            textPanelX, textPanelY, displayWidth - textPanelX, textCellHeight, COLOR_BLACK);
        PresentResults(results, labels, textStartX, textStartY);
        DrawDetectionBoxes(results, imageStartX, imageStartY, imageWidth, imageHeight, downscale);

#if VERIFY_TEST_OUTPUT
        for (size_t outputIndex = 0; outputIndex < model.GetNumOutputs(); ++outputIndex) {
            const auto output = model.GetOutputTensor(outputIndex);
            DumpTensorData(static_cast<const uint8_t*>(output->GetData()), output->Bytes());
        }
#endif

        profiler.PrintProfilingResult();
    }

    return true;
}

} /* namespace arm::app */
