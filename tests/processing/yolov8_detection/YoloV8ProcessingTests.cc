/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/yolov8_detection/YoloV8PostProcessing.hpp"
#include "mlek/use_case/yolov8_detection/YoloV8PreProcessing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
using namespace arm::app;
using namespace arm::app::fwk::iface;

void Require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class Tensor : public TensorIface {
public:
    Tensor(std::vector<size_t> shape, TensorType type, QuantParams quant = {}) :
        m_shape(std::move(shape)), m_type(type), m_quant(quant)
    {
        size_t elements = 1;
        for (const auto dimension : m_shape) {
            elements *= dimension;
        }
        m_storage.resize((elements * GetTensorDataTypeSize(type) + 3) / 4);
    }
    void* GetData() override
    { return m_storage.data(); }
    size_t Bytes() override
    { return GetNumElements() * GetTensorDataTypeSize(m_type); }
    size_t GetNumElements() override
    {
        size_t count = 1;
        for (const auto dimension : m_shape) {
            count *= dimension;
        }
        return count;
    }
    TensorType Type() override
    { return m_type; }
    TensorLayout Layout() override
    { return TensorLayout::NHWC; }
    std::vector<size_t> Shape() override
    { return m_shape; }
    QuantParams GetQuantParams() override
    { return m_quant; }
    void SetQuant(QuantParams quant)
    { m_quant = quant; }

private:
    std::vector<size_t> m_shape;
    TensorType m_type;
    QuantParams m_quant;
    std::vector<uint32_t> m_storage;
};

void CheckInputQuantization(bool inputIsGrayscale)
{
    std::vector<uint8_t> pixels(256 * 3);
    for (size_t value = 0; value < 256; ++value) {
        std::fill_n(pixels.begin() + value * 3, 3, static_cast<uint8_t>(value));
    }
    for (const auto type : {TensorType::INT8, TensorType::UINT8}) {
        auto tensor = std::make_shared<Tensor>(std::vector<size_t>{1, 16, 16, 1}, type);
        YoloV8PreProcess process(tensor, false, inputIsGrayscale);
        // Exercise direct conversion, lookup conversion, invalidation and direct conversion again.
        for (const auto quant : {QuantParams{1.0F / 255.0F, -128},
                                 QuantParams{0.00378316035F, -128},
                                 QuantParams{0.007F, 23},
                                 QuantParams{1.0F / 255.0F, -128}}) {
            tensor->SetQuant(quant);
            Require(process.DoPreProcess(pixels.data(), pixels.size()), "input conversion failed");
            for (size_t value = 0; value < 256; ++value) {
                const long minimum  = type == TensorType::INT8 ? -128 : 0;
                const long maximum  = type == TensorType::INT8 ? 127 : 255;
                const long expected = std::clamp(
                    std::lround((static_cast<float>(value) / 255.0F) / quant.scale) + quant.offset,
                    minimum,
                    maximum);
                const int actual = type == TensorType::INT8
                                       ? static_cast<int8_t*>(tensor->GetData())[value]
                                       : static_cast<uint8_t*>(tensor->GetData())[value];
                Require(actual == expected, "input quantization changed an integer code");
            }
        }
        tensor->SetQuant({std::numeric_limits<float>::quiet_NaN(), 0});
        Require(!process.DoPreProcess(pixels.data(), pixels.size()), "NaN scale accepted");
    }
    auto tensor = std::make_shared<Tensor>(std::vector<size_t>{1, 2, 2, 1}, TensorType::FP32);
    const uint8_t rgb[] = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255};
    YoloV8PreProcess process(tensor);
    Require(process.DoPreProcess(rgb, sizeof(rgb)), "RGB-to-gray conversion failed");
    const auto* values = static_cast<float*>(tensor->GetData());
    for (size_t index = 0; index < 4; ++index) {
        const int expected[] = {76, 150, 29, 255};
        Require(std::abs(values[index] - expected[index] / 255.0F) < 1e-7F,
                "RGB-to-gray rounding differs from Pillow");
    }
}

void CheckRgbModelKeepsAllChannels()
{
    auto tensor = std::make_shared<Tensor>(
        std::vector<size_t>{1, 2, 2, 3}, TensorType::INT8, QuantParams{1.0F / 255.0F, -128});
    const uint8_t rgb[] = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255};
    YoloV8PreProcess process(tensor, false, true);
    Require(process.DoPreProcess(rgb, sizeof(rgb)), "RGB model preprocessing failed");
    const auto* actual = static_cast<int8_t*>(tensor->GetData());
    for (size_t index = 0; index < sizeof(rgb); ++index) {
        Require(actual[index] == static_cast<int>(rgb[index]) - 128,
                "grayscale flag changed a three-channel model input");
    }
}

std::vector<std::shared_ptr<TensorIface>> MakeHeads(bool quantized)
{
    std::vector<std::shared_ptr<TensorIface>> heads;
    for (const size_t size : {8, 4, 2}) {
        auto tensor = std::make_shared<Tensor>(std::vector<size_t>{1, size, size, 34},
                                               quantized ? TensorType::INT8 : TensorType::FP32,
                                               QuantParams{0.25F, 0});
        std::vector<float> values(size * size * 34, -8.0F);
        for (size_t cell = 0; cell < size * size; ++cell) {
            for (size_t side = 0; side < 4; ++side) {
                values[cell * 34 + side * 8 + 1] = 8.0F;
            }
        }
        if (size == 8) {
            const size_t cells[] = {9, 12, 15, 33, 36, 39};
            for (size_t index = 0; index < 6; ++index) {
                values[cells[index] * 34 + 32] = 4.0F - index * 0.25F;
                // Multiple passing classes still yield one selected class per location.
                values[cells[index] * 34 + 33] = 1.0F;
            }
        }
        if (quantized) {
            auto* destination = static_cast<int8_t*>(tensor->GetData());
            for (size_t index = 0; index < values.size(); ++index) {
                destination[index] = static_cast<int8_t>(std::lround(values[index] / 0.25F));
            }
        } else {
            std::memcpy(tensor->GetData(), values.data(), values.size() * sizeof(float));
        }
        heads.push_back(tensor);
    }
    heads.push_back(std::make_shared<Tensor>(std::vector<size_t>{1, 1, 84}, TensorType::FP32));
    return heads;
}

void CheckMultipleRotatedDetections()
{
    const YoloV8PostProcessParams params{64, 64, 2, 5, 0.25F, 0.45F, true, 8, true};
    std::vector<yolov8_detection::DetectionResult> reference;
    for (const bool quantized : {false, true}) {
        auto heads = MakeHeads(quantized);
        std::vector<yolov8_detection::DetectionResult> results;
        YoloV8NewPostProcess process(heads, results, params, false);
        Require(process.DecodeDetections(), "multi-target OBB decode failed");
        process.SuppressDetections();
        Require(results.size() == 5, "expected five detections after rotated NMS");
        for (size_t index = 0; index < results.size(); ++index) {
            Require(results[index].classId == 0, "more than one class emitted per location");
            if (index > 0) {
                Require(results[index - 1].score >= results[index].score, "scores not sorted");
            }
            if (quantized) {
                Require(std::abs(results[index].score - reference[index].score) < 1e-6F,
                        "quantized probability lookup differs from FP32");
                Require(std::abs(results[index].width - reference[index].width) < 1e-4F,
                        "quantized DFL lookup differs from FP32");
            }
        }
        if (!quantized) {
            reference = results;
        }
    }
    auto heads = MakeHeads(false);
    std::vector<yolov8_detection::DetectionResult> results;
    auto general          = params;
    general.maxDetections = 6;
    YoloV8NewPostProcess sixDetections(heads, results, general);
    Require(sixDetections.DoPostProcess(), "generic decoder rejected a six-result limit");
    Require(results.size() == 6, "generic decoder did not honor the configured limit");
}

void CheckLimitAfterSuppression()
{
    // Six high-scoring overlapping boxes must not hide a later independent detection.
    auto tensor  = std::make_shared<Tensor>(std::vector<size_t>{1, 8, 6}, TensorType::FP32);
    auto* values = static_cast<float*>(tensor->GetData());
    for (size_t index = 0; index < 8; ++index) {
        const float box[] = {index < 6 ? 12.0F : 50.0F,
                             12.0F,
                             8.0F,
                             8.0F,
                             0.99F - index * 0.01F,
                             index == 7 ? 0.95F : 0.0F};
        std::copy_n(box, 6, values + index * 6);
    }
    std::vector<yolov8_detection::DetectionResult> results;
    YoloV8PostProcess process(tensor, results, {64, 64, 2, 5, 0.25F, 0.45F});
    Require(process.DoPostProcess(), "single-output decode failed");
    Require(results.size() == 3, "candidate truncation or class-aware NMS is incorrect");
}

void CheckExplicitBoxCoordinates()
{
    auto tensor = std::make_shared<Tensor>(std::vector<size_t>{1, 6, 2}, TensorType::FP32);
    // An out-of-image, low-confidence box must not change the coordinate units
    // of the valid detection. The alif COCO model emits such normalized outliers.
    const float values[] = {
        0.5F, 2.01F, 0.4F, 0.4F, 0.2F, 0.1F, 0.2F, 0.1F, 0.9F, 0.01F, 0.01F, 0.01F};
    std::memcpy(tensor->GetData(), values, sizeof(values));
    std::vector<yolov8_detection::DetectionResult> results;
    YoloV8PostProcessParams params{256, 256, 2, 20, 0.1F, 0.45F};
    params.boxCoordinates = BoxCoordinateFormat::NORMALIZED;
    YoloV8PostProcess normalized(tensor, results, params);
    Require(normalized.DoPostProcess() && results.size() == 1,
            "explicit normalized-coordinate decode failed");
    Require(std::fabs(results[0].centerX - 128.0F) < 0.001F &&
                std::fabs(results[0].width - 51.2F) < 0.001F,
            "normalized outlier caused pixel-space decoding");

    // Small pixel-space boxes must also obey an explicit model contract.
    params.boxCoordinates = BoxCoordinateFormat::PIXELS;
    YoloV8PostProcess pixels(tensor, results, params);
    Require(pixels.DoPostProcess() && results.size() == 1,
            "explicit pixel-coordinate decode failed");
    Require(std::fabs(results[0].centerX - 0.5F) < 0.001F &&
                std::fabs(results[0].width - 0.2F) < 0.001F,
            "explicit pixel coordinates were scaled");
}

void CheckAllGestureClasses()
{
    // The full HaGRID model has 33 classes and a decoded [1, 37, anchors] output.
    // Overlapping boxes in the first and last classes must survive class-aware NMS.
    auto tensor  = std::make_shared<Tensor>(std::vector<size_t>{1, 37, 2}, TensorType::FP32);
    auto* values = static_cast<float*>(tensor->GetData());
    std::fill_n(values, 74, 0.0F);
    for (size_t anchor = 0; anchor < 2; ++anchor) {
        values[anchor]     = 0.5F;
        values[2 + anchor] = 0.5F;
        values[4 + anchor] = 0.2F;
        values[6 + anchor] = 0.2F;
    }
    values[(4 + 32) * 2] = 0.9F;
    values[4 * 2 + 1]    = 0.8F;
    std::vector<yolov8_detection::DetectionResult> results;
    YoloV8PostProcessParams params{320, 320, 33, 20, 0.25F, 0.45F};
    params.boxCoordinates = BoxCoordinateFormat::NORMALIZED;
    YoloV8PostProcess process(tensor, results, params);
    Require(process.DoPostProcess(), "33-class gesture output decode failed");
    Require(results.size() == 2 && results[0].classId == 32 && results[1].classId == 0,
            "full gesture class range or class-aware NMS failed");
    Require(std::abs(results[0].centerX - 160.0F) < 0.001F,
            "full gesture normalized boxes were not converted to model pixels");
}

void CheckRotatedSuppressionAndClassTies()
{
    auto heads   = MakeHeads(false);
    auto* values = static_cast<float*>(heads[0]->GetData());
    // Suppress the ordinary fixture, then put two long thin crossing boxes nearby.
    for (size_t cell = 0; cell < 64; ++cell) {
        values[cell * 34 + 32] = -8.0F;
        values[cell * 34 + 33] = -8.0F;
    }
    for (const size_t cell : {27, 28}) {
        for (size_t side = 0; side < 4; ++side) {
            for (size_t bin = 0; bin < 8; ++bin) {
                values[cell * 34 + side * 8 + bin] =
                    bin == (side % 2 == 0 ? 6U : 1U) ? 8.0F : -8.0F;
            }
        }
        // Both scores saturate to one. The upstream rule keeps the first class.
        values[cell * 34 + 32] = 20.0F;
        values[cell * 34 + 33] = 30.0F;
    }
    auto* angles = static_cast<float*>(heads[3]->GetData());
    angles[27]   = 0.0F;
    angles[28]   = -20.0F;
    std::vector<yolov8_detection::DetectionResult> results;
    YoloV8NewPostProcess process(heads, results, {64, 64, 2, 5, 0.25F, 0.45F, true, 8, true});
    Require(process.DoPostProcess(), "crossing OBB decode failed");
    Require(results.size() == 2, "rotated NMS incorrectly suppressed crossing thin boxes");
    Require(results[0].classId == 0 && results[1].classId == 0,
            "Sigmoid saturation changed the selected class");
    angles[28] = 0.0F;
    Require(process.DoPostProcess(), "parallel OBB decode failed");
    Require(results.size() == 1, "rotated NMS failed to suppress overlapping parallel boxes");
}
} // namespace

int main()
{
    try {
        CheckInputQuantization(false);
        CheckInputQuantization(true);
        CheckRgbModelKeepsAllChannels();
        CheckMultipleRotatedDetections();
        CheckLimitAfterSuppression();
        CheckExplicitBoxCoordinates();
        CheckAllGestureClasses();
        CheckRotatedSuppressionAndClassTies();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    std::puts("YOLO processing regression checks passed");
    return 0;
}
