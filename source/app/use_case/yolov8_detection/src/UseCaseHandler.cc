/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "UseCaseHandler.hpp"
#include "BenchmarkProfiler.hpp"
#include "DetectionDrawing.hpp"

#include "UseCaseCommonUtils.hpp"
#include "hal.h"
#include "mlek/common/ImageUtils.hpp"
#include "mlek/log/log_macros.h"
#include "mlek/use_case/yolov8_detection/YoloV8PostProcessing.hpp"
#include "mlek/use_case/yolov8_detection/YoloV8PreProcessing.hpp"
#include "sample_files.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace arm::app {

namespace {

#if YOLOV8_DUMP_RAW_OUTPUTS
    void DumpRawOutputs(fwk::iface::Model& model)
    {
        for (size_t index = 0; index < model.GetNumOutputs(); ++index) {
            const auto tensor      = model.GetOutputTensor(index);
            const auto quantParams = tensor->GetQuantParams();
            const auto* data       = static_cast<const uint8_t*>(tensor->GetData());
            const size_t bytes     = tensor->Bytes();
            printf("RAW_OUTPUT_BEGIN index=%zu type=%d bytes=%zu scale=%.9g zero=%d shape=",
                   index,
                   static_cast<int>(tensor->Type()),
                   bytes,
                   static_cast<double>(quantParams.scale),
                   static_cast<int>(quantParams.offset));
            for (const size_t dimension : tensor->Shape()) {
                printf("%zu,", dimension);
            }
            printf("\n");
            for (size_t offset = 0; offset < bytes; ++offset) {
                if (offset % 32 == 0) {
                    printf("RAW_OUTPUT_DATA ");
                }
                printf("%02x", static_cast<unsigned int>(data[offset]));
                if (offset % 32 == 31 || offset + 1 == bytes) {
                    printf("\n");
                }
            }
            printf("RAW_OUTPUT_END index=%zu\n", index);
        }
    }
#endif

#if YOLOV8_PROFILE_STAGES
    bool ReadCpuCycles(const pmu_counters& counters, uint64_t& cycles)
    {
        if (!counters.initialised) {
            return false;
        }
        for (uint32_t index = 0; index < counters.num_counters; ++index) {
            if (std::strcmp(counters.counters[index].name, "CPU TOTAL") == 0) {
                cycles = counters.counters[index].value;
                return true;
            }
        }
        return false;
    }

    bool RecordCpuCycles(Profiler& profiler, const char* name, uint64_t start, uint64_t finish)
    {
        pmu_counters first{};
        first.initialised      = true;
        first.num_counters     = 1;
        first.counters[0]      = {start, "CPU TOTAL", "cycles"};
        pmu_counters last      = first;
        last.counters[0].value = finish;
        return profiler.RecordProfiling(first, last, name);
    }

    bool RecordStages(Profiler& profiler, const std::array<pmu_counters, 4>& timestamps)
    {
        std::array<uint64_t, 4> cycles{};
        for (size_t index = 0; index < timestamps.size(); ++index) {
            if (!ReadCpuCycles(timestamps[index], cycles[index])) {
                return false;
            }
        }
        return RecordCpuCycles(profiler, "Preprocess + Invoke", cycles[0], cycles[2]) &&
               profiler.RecordProfiling(timestamps[0], timestamps[2], "Preprocess + Inference") &&
               RecordCpuCycles(
                   profiler, "Preprocess + Inference + Postprocess", cycles[0], cycles[3]) &&
               RecordCpuCycles(profiler, "Preprocess", cycles[0], cycles[1]) &&
               RecordCpuCycles(profiler, "Inference", cycles[1], cycles[2]) &&
               RecordCpuCycles(profiler, "Postprocess", cycles[2], cycles[3]) &&
               RecordCpuCycles(profiler, "Inference + Postprocess", cycles[1], cycles[3]);
    }

    bool RecordOperators(Profiler& profiler, const BenchmarkOpProfiler& opProfiler)
    {
        using Operation        = BenchmarkOpProfiler::Operation;
        const auto& quantize   = opProfiler.Get(Operation::QUANTIZE);
        const auto& ethosU     = opProfiler.Get(Operation::ETHOS_U);
        const auto& dequantize = opProfiler.Get(Operation::DEQUANTIZE);
        info("Model operator calls: QUANTIZE=%" PRIu32 " ethos-u=%" PRIu32 " DEQUANTIZE=%" PRIu32
             "\n",
             quantize.calls,
             ethosU.calls,
             dequantize.calls);
        return (quantize.calls == 0 ||
                RecordCpuCycles(profiler, "Model QUANTIZE", 0, quantize.cycles)) &&
               (ethosU.calls == 0 ||
                RecordCpuCycles(profiler, "Model ethos-u (CPU elapsed)", 0, ethosU.cycles)) &&
               (dequantize.calls == 0 ||
                RecordCpuCycles(profiler, "Model DEQUANTIZE total", 0, dequantize.cycles));
    }
#endif

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

            info("%zu: class=%d label=%s score=%f cx=%f cy=%f w=%f h=%f\n",
                 index,
                 result.classId,
                 GetLabel(labels, result.classId),
                 result.score,
                 result.centerX,
                 result.centerY,
                 result.width,
                 result.height);
            if (YOLOV8_OBB) {
                info("   obb: cx=%f cy=%f w=%f h=%f angle_rad=%f\n",
                     result.centerX,
                     result.centerY,
                     result.width,
                     result.height,
                     result.angleRadians);
            }
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

        if ((shape[1] == 3 && shape[3] != 3) || (shape[1] == 1 && shape[3] != 1 && shape[3] != 3)) {
            imageHeight = static_cast<uint32_t>(shape[2]);
            imageWidth  = static_cast<uint32_t>(shape[3]);
            return true;
        }

        if ((shape[3] == 3 && shape[1] != 3) || (shape[3] == 1 && shape[1] != 1 && shape[1] != 3)) {
            imageHeight = static_cast<uint32_t>(shape[1]);
            imageWidth  = static_cast<uint32_t>(shape[2]);
            return true;
        }

        return false;
    }

} /* anonymous namespace */

#if YOLOV8_PROFILE_STAGES
uint64_t ReadBenchmarkCpuCycles()
{
    pmu_counters counters{};
    hal_pmu_get_cpu_counters(&counters);
    uint64_t cycles = 0;
    ReadCpuCycles(counters, cycles);
    return cycles;
}
#endif

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
        printf_err("Expected RGB or grayscale NCHW/NHWC input\n");
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

    /* Native NHWC logical axes can explicitly use shape inference. ExecuTorch's
     * dim_order otherwise describes physical storage of NCHW logical axes. */
#if defined(MLEK_FWK_EXECUTORCH) && !YOLOV8_INFER_LAYOUT_FROM_SHAPE
    constexpr bool useTensorLayout = true;
#else
    constexpr bool useTensorLayout = false;
#endif
    YoloV8PreProcess preProcess{inputTensor, useTensorLayout, YOLOV8_INPUT_IS_GRAYSCALE != 0};

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
                                   YOLOV8_NMS_THRESHOLD,
                                   YOLOV8_DFL_BINS,
                                   YOLOV8_OBB != 0};
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
        postProcess =
            std::make_unique<YoloV8NewPostProcess>(outputTensors, results, params, useTensorLayout);
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
    const uint32_t textPanelWidth = displayWidth - textStartX;

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

#if YOLOV8_PROFILE_STAGES
        std::array<pmu_counters, 4> stageTimestamps{};
        auto& opProfiler = ctx.Get<BenchmarkOpProfiler&>("benchmark_op_profiler");
        opProfiler.Reset();
        hal_pmu_reset();
        const uint64_t probeStart = ReadBenchmarkCpuCycles();
        const uint64_t probeEnd   = ReadBenchmarkCpuCycles();
        hal_pmu_get_counters(&stageTimestamps[0]);
#endif
        if (!preProcess.DoPreProcess(image, get_sample_img_total_bytes())) {
#if YOLOV8_PROFILE_STAGES
            hal_pmu_final();
            opProfiler.Stop();
#endif
            printf_err("Pre-processing failed\n");
            return false;
        }

#if YOLOV8_PROFILE_STAGES
        hal_pmu_get_cpu_counters(&stageTimestamps[1]);
        const bool inferenceSuccessful = model.RunInference();
        hal_pmu_get_counters(&stageTimestamps[2]);
        const bool postProcessSuccessful = inferenceSuccessful && postProcess->DoPostProcess();
        hal_pmu_get_cpu_counters(&stageTimestamps[3]);
        hal_pmu_final();
        opProfiler.Stop();
        info("Timer read interval: %" PRIu64 " cycles\n", probeEnd - probeStart);
        if (model.ContainsEthosUOperator() &&
            opProfiler.Get(BenchmarkOpProfiler::Operation::ETHOS_U).calls == 0) {
            printf_err("Model operator profiler did not observe the Ethos-U call\n");
            return false;
        }
        if (!RecordStages(profiler, stageTimestamps)) {
            printf_err("Failed to record benchmark stages\n");
            return false;
        }
        if (!RecordOperators(profiler, opProfiler)) {
            printf_err("Failed to record operator timings\n");
            return false;
        }
#elif YOLOV8_PROFILE_POSTPROCESS
        if (!profiler.StartProfiling("Inference + Postprocess")) {
            return false;
        }
        const bool inferenceSuccessful   = model.RunInference();
        const bool postProcessSuccessful = inferenceSuccessful && postProcess->DoPostProcess();
        if (!profiler.StopProfiling()) {
            return false;
        }
#else
        const bool inferenceSuccessful   = RunInference(model, profiler);
        const bool postProcessSuccessful = inferenceSuccessful && postProcess->DoPostProcess();
#endif

        if (!inferenceSuccessful) {
            printf_err("Inference failed\n");
            return false;
        }

        if (!postProcessSuccessful) {
            printf_err("Post-processing failed\n");
            return false;
        }

        PrintResults(results, labels);
#if YOLOV8_DUMP_RAW_OUTPUTS
        DumpRawOutputs(model);
#endif
        hal_display_show_box(textStartX, textStartY, textPanelWidth, 16, COLOR_BLACK);
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
