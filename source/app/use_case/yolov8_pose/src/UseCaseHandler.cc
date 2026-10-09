/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "UseCaseHandler.hpp"

#include "UseCaseCommonUtils.hpp"
#include "hal.h"
#include "mlek/common/ImageUtils.hpp"
#include "mlek/log/log_macros.h"
#include "mlek/use_case/yolov8_pose/YoloV8PosePostProcessing.hpp"
#include "mlek/use_case/yolov8_pose/YoloV8PosePreProcessing.hpp"
#include "sample_files.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace arm::app {
namespace {

    struct ImageDimensions {
        uint32_t width{0};
        uint32_t height{0};
    };

    bool GetImageDimensions(const std::vector<size_t>& shape, ImageDimensions& dimensions)
    {
        if (shape.size() != 4 || shape[0] != 1) {
            return false;
        }

        if (shape[1] == 3 && shape[3] != 3) {
            dimensions.height = shape[2];
            dimensions.width  = shape[3];
            return true;
        }

        if (shape[3] == 3 && shape[1] != 3) {
            dimensions.height = shape[1];
            dimensions.width  = shape[2];
            return true;
        }

        return false;
    }

    const char* GetLabel(const std::vector<std::string>& labels, int classId)
    {
        if (classId < 0 || static_cast<size_t>(classId) >= labels.size()) {
            return "unknown";
        }
        return labels[static_cast<size_t>(classId)].c_str();
    }

    void DrawPoseResults(const std::vector<yolov8_pose::PoseResult>& results,
                         uint32_t startX,
                         uint32_t startY,
                         uint32_t downscale)
    {
        static constexpr std::array<std::array<size_t, 2>, 20> handEdges{{
            {{0, 1}},   {{1, 2}},   {{2, 3}},  {{3, 4}},   {{0, 5}},   {{5, 6}},   {{6, 7}},
            {{7, 8}},   {{0, 9}},   {{9, 10}}, {{10, 11}}, {{11, 12}}, {{0, 13}},  {{13, 14}},
            {{14, 15}}, {{15, 16}}, {{0, 17}}, {{17, 18}}, {{18, 19}}, {{19, 20}},
        }};
        constexpr uint32_t boxThickness = 1;
        constexpr uint32_t keypointSize = 3;

        const auto drawLine = [](int x0, int y0, int x1, int y1) {
            const int deltaX = std::abs(x1 - x0);
            const int stepX  = x0 < x1 ? 1 : -1;
            const int deltaY = -std::abs(y1 - y0);
            const int stepY  = y0 < y1 ? 1 : -1;
            int error        = deltaX + deltaY;

            while (true) {
                hal_display_show_box(
                    static_cast<uint32_t>(x0), static_cast<uint32_t>(y0), 1, 1, COLOR_YELLOW);
                if (x0 == x1 && y0 == y1) {
                    break;
                }
                const int twiceError = 2 * error;
                if (twiceError >= deltaY) {
                    error += deltaY;
                    x0 += stepX;
                }
                if (twiceError <= deltaX) {
                    error += deltaX;
                    y0 += stepY;
                }
            }
        };

        for (const auto& result : results) {
            const uint32_t x      = startX + static_cast<uint32_t>(result.x) / downscale;
            const uint32_t y      = startY + static_cast<uint32_t>(result.y) / downscale;
            const uint32_t width  = static_cast<uint32_t>(result.width) / downscale;
            const uint32_t height = static_cast<uint32_t>(result.height) / downscale;

            if (width > boxThickness && height > boxThickness) {
                hal_display_show_box(x, y, width, boxThickness, COLOR_GREEN);
                hal_display_show_box(
                    x, y + height - boxThickness, width, boxThickness, COLOR_GREEN);
                hal_display_show_box(x, y, boxThickness, height, COLOR_GREEN);
                hal_display_show_box(
                    x + width - boxThickness, y, boxThickness, height, COLOR_GREEN);
            }

            for (const auto& edge : handEdges) {
                const auto& start = result.keypoints[edge[0]];
                const auto& end   = result.keypoints[edge[1]];
                if (start.score < YOLOV8_POSE_KEYPOINT_THRESHOLD ||
                    end.score < YOLOV8_POSE_KEYPOINT_THRESHOLD) {
                    continue;
                }
                drawLine(static_cast<int>(startX) +
                             static_cast<int>(std::lround(start.x)) / static_cast<int>(downscale),
                         static_cast<int>(startY) +
                             static_cast<int>(std::lround(start.y)) / static_cast<int>(downscale),
                         static_cast<int>(startX) +
                             static_cast<int>(std::lround(end.x)) / static_cast<int>(downscale),
                         static_cast<int>(startY) +
                             static_cast<int>(std::lround(end.y)) / static_cast<int>(downscale));
            }

            for (const auto& keypoint : result.keypoints) {
                if (keypoint.score < YOLOV8_POSE_KEYPOINT_THRESHOLD) {
                    continue;
                }

                const uint32_t keypointX =
                    startX + static_cast<uint32_t>(std::lround(keypoint.x)) / downscale;
                const uint32_t keypointY =
                    startY + static_cast<uint32_t>(std::lround(keypoint.y)) / downscale;
                hal_display_show_box(
                    keypointX, keypointY, keypointSize, keypointSize, COLOR_YELLOW);
            }
        }
    }

    void PrintResults(const std::vector<yolov8_pose::PoseResult>& results,
                      const std::vector<std::string>& labels,
                      const char* imageName,
                      uint32_t imageWidth,
                      uint32_t imageHeight)
    {
        info("Hand Pose Results: %zu detections after NMS image=%s\n", results.size(), imageName);
        for (size_t index = 0; index < results.size(); ++index) {
            const auto& result = results[index];
            info("hand[%zu] score=%f box=[%d, %d, %d, %d] source=%" PRIu32 "x%" PRIu32
                 " class=%d label=%s\n",
                 index,
                 result.score,
                 result.x,
                 result.y,
                 result.x + result.width,
                 result.y + result.height,
                 imageWidth,
                 imageHeight,
                 result.classId,
                 GetLabel(labels, result.classId));

            for (size_t keypoint = 0; keypoint < result.keypoints.size(); ++keypoint) {
                const auto& point = result.keypoints[keypoint];
                if (point.score >= YOLOV8_POSE_KEYPOINT_THRESHOLD) {
                    info("  kp[%zu] x=%f y=%f score=%f\n", keypoint, point.x, point.y, point.score);
                }
            }
        }
    }

    void PresentResults(const std::vector<yolov8_pose::PoseResult>& results,
                        uint32_t startX,
                        uint32_t startY)
    {
        char line[32]{};
        hal_display_set_text_color(COLOR_GREEN);
        const int length = std::snprintf(line, sizeof(line), "Poses: %zu", results.size());
        if (length > 0) {
            hal_display_show_text(line,
                                  std::min(static_cast<size_t>(length), sizeof(line) - 1),
                                  startX,
                                  startY,
                                  false);
        }
    }

} /* anonymous namespace */

bool YoloV8PoseHandler(ApplicationContext& ctx)
{
    auto& profiler     = ctx.Get<Profiler&>("profiler");
    auto& model        = ctx.Get<fwk::iface::Model&>("model");
    const auto& labels = ctx.Get<const std::vector<std::string>&>("labels");

    if (!model.IsInited()) {
        printf_err("Model is not initialised\n");
        return false;
    }

    const size_t numOutputs = model.GetNumOutputs();
    if (model.GetNumInputs() != 1 || (numOutputs != 1 && numOutputs != 4)) {
        printf_err("YOLOv8 pose model must have one input and one or four outputs\n");
        return false;
    }
#if YOLOV8_POSE_EXPECTED_NUM_OUTPUTS != 0
    if (numOutputs != YOLOV8_POSE_EXPECTED_NUM_OUTPUTS) {
        printf_err("YOLOv8 pose output topology mismatch: expected %d outputs, got %zu\n",
                   YOLOV8_POSE_EXPECTED_NUM_OUTPUTS,
                   numOutputs);
        return false;
    }
#endif

    if (labels.size() != YOLOV8_POSE_NUM_CLASSES) {
        printf_err("YOLOv8 pose label count does not match the configured class count\n");
        return false;
    }

    auto inputTensor = model.GetInputTensor(0);
    std::vector<std::shared_ptr<fwk::iface::TensorIface>> outputTensors;
    outputTensors.reserve(model.GetNumOutputs());
    for (size_t output = 0; output < model.GetNumOutputs(); ++output) {
        outputTensors.push_back(model.GetOutputTensor(output));
    }
    ImageDimensions dimensions;
    if (inputTensor == nullptr ||
        std::any_of(outputTensors.begin(),
                    outputTensors.end(),
                    [](const auto& tensor) { return tensor == nullptr; }) ||
        !GetImageDimensions(inputTensor->Shape(), dimensions)) {
        printf_err("Invalid YOLOv8 pose input or output tensor\n");
        return false;
    }
    if (!ValidateYoloV8PoseOutputTensors(outputTensors, YOLOV8_POSE_NUM_CLASSES)) {
        printf_err("Invalid YOLOv8 pose packed/split output tensor contract\n");
        return false;
    }

    info("YOLOv8 pose output contract: %s, %s coordinates\n",
         numOutputs == 1 ? "packed" : "split",
         YOLOV8_POSE_COORDINATES_NORMALIZED ? "normalized" : "pixel");

    if (get_sample_img_width() != dimensions.width ||
        get_sample_img_height() != dimensions.height) {
        printf_err("Generated image dimensions do not match model input\n");
        return false;
    }

    auto inputLayout = fwk::iface::TensorLayout::INVALID;
#if defined(MLEK_FWK_TFLM)
    /* TFLite tensors use their logical order; this export has an NCHW input.
     * ExecuTorch instead exposes a separate physical dim_order. */
    inputLayout = inputTensor->Shape()[1] == 3 ? fwk::iface::TensorLayout::NCHW
                                               : fwk::iface::TensorLayout::NHWC;
#endif
    YoloV8PosePreProcess preProcess{inputTensor, inputLayout};
    std::vector<yolov8_pose::PoseResult> results;
    YoloV8PosePostProcess postProcess{outputTensors,
                                      results,
                                      {dimensions.width,
                                       dimensions.height,
                                       YOLOV8_POSE_NUM_CLASSES,
                                       YOLOV8_POSE_MAX_DETECTIONS,
                                       YOLOV8_POSE_SCORE_THRESHOLD,
                                       YOLOV8_POSE_NMS_THRESHOLD,
                                       YOLOV8_POSE_COORDINATES_NORMALIZED}};

    constexpr uint32_t imageStartX   = 10;
    constexpr uint32_t imageStartY   = 35;
    constexpr uint32_t textStartY    = 40;
    constexpr uint32_t displayWidth  = 320;
    constexpr uint32_t displayHeight = 240;
    constexpr uint32_t downscale     = YOLOV8_POSE_DISPLAY_DOWNSCALE;
    static_assert(downscale > 0, "Display downscale factor must be greater than zero");

    const uint32_t displayedImageWidth  = dimensions.width / downscale;
    const uint32_t displayedImageHeight = dimensions.height / downscale;
    const uint32_t textStartX           = imageStartX + displayedImageWidth + 10;
    if (textStartX >= displayWidth || imageStartY + displayedImageHeight > displayHeight) {
        printf_err("Displayed pose image does not fit the Corstone-320 HDLCD\n");
        return false;
    }

    for (uint32_t index = 0; index < get_sample_n_elements(); ++index) {
#ifdef INTERACTIVE_MODE
        AwaitUserInput();
#endif

        const uint8_t* image = get_sample_data_ptr(index);
        if (image == nullptr) {
            printf_err("Failed to get pose input image\n");
            return false;
        }

        info("Running pose image %" PRIu32 ": %s\n", index, get_sample_data_filename(index));
        hal_display_clear(COLOR_BLACK);
        hal_display_show_image(
            image, dimensions.width, dimensions.height, 3, imageStartX, imageStartY, downscale);

        if (!profiler.StartProfiling("Pre-processing")) {
            return false;
        }
        const bool preProcessSuccessful =
            preProcess.DoPreProcess(image, get_sample_img_total_bytes());
        const bool preProfileSuccessful = profiler.StopProfiling();
        if (!preProcessSuccessful || !preProfileSuccessful) {
            printf_err("YOLOv8 pose pre-processing failed\n");
            return false;
        }

        for (uint32_t warmup = 0; warmup < YOLOV8_POSE_PMU_WARMUP_RUNS; ++warmup) {
            if (!model.RunInference()) {
                printf_err("YOLOv8 pose warm-up inference failed\n");
                return false;
            }
        }
        for (uint32_t measured = 0; measured < YOLOV8_POSE_PMU_MEASURED_RUNS; ++measured) {
            if (!profiler.StartProfiling("Inference")) {
                return false;
            }
            const bool inferenceSuccessful = model.RunInference();
            const bool profileSuccessful   = profiler.StopProfiling();
            if (!inferenceSuccessful || !profileSuccessful) {
                printf_err("YOLOv8 pose inference failed\n");
                return false;
            }
        }

        if (!profiler.StartProfiling("Post-processing")) {
            return false;
        }
        const bool postProcessSuccessful = postProcess.DoPostProcess();
        const bool postProfileSuccessful = profiler.StopProfiling();
        if (!postProcessSuccessful || !postProfileSuccessful) {
            printf_err("YOLOv8 pose post-processing failed\n");
            return false;
        }

        profiler.PrintProfilingResult();

        PrintResults(
            results, labels, get_sample_data_filename(index), dimensions.width, dimensions.height);
        PresentResults(results, textStartX, textStartY);
        DrawPoseResults(results, imageStartX, imageStartY, downscale);

#if VERIFY_TEST_OUTPUT
        for (const auto& outputTensor : outputTensors) {
            DumpTensor(outputTensor);
        }
#endif
    }

    return true;
}

} /* namespace arm::app */
