/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "QrCodePreProcessing.hpp"
#include "mlek/use_case/yolov8_detection/YoloV8PostProcessing.hpp"
#include "mlek/use_case/yolov8_detection/YoloV8PreProcessing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <numeric>

using namespace arm::app;
using namespace arm::app::fwk::iface;

static_assert(sizeof(yolov8_detection::DetectionResult) == 28);

class MemoryTensor : public TensorIface {
public:
    MemoryTensor(std::vector<size_t> shape,
                 TensorType type,
                 TensorLayout layout,
                 QuantParams quant = {0.125F, 0}) :
        m_shape{std::move(shape)}, m_type{type}, m_layout{layout}, m_quant{quant},
        m_data(GetNumElements() * GetTensorDataTypeSize(type))
    {}
    void* GetData() override
    { return m_data.data(); }
    size_t Bytes() override
    { return m_data.size(); }
    size_t GetNumElements() override
    { return std::accumulate(m_shape.begin(), m_shape.end(), size_t{1}, std::multiplies<>()); }
    TensorType Type() override
    { return m_type; }
    TensorLayout Layout() override
    { return m_layout; }
    std::vector<size_t> Shape() override
    { return m_shape; }
    QuantParams GetQuantParams() override
    { return m_quant; }
    void SetQuantParams(QuantParams quant)
    { m_quant = quant; }
    float Value(size_t index)
    {
        if (m_type == TensorType::FP32) {
            return static_cast<float*>(GetData())[index];
        }
        const int raw = m_type == TensorType::INT8 ? static_cast<int8_t*>(GetData())[index]
                                                   : static_cast<uint8_t*>(GetData())[index];
        return (static_cast<float>(raw) - static_cast<float>(m_quant.offset)) * m_quant.scale;
    }
    void Set(size_t index, float value)
    {
        if (m_type == TensorType::FP32) {
            static_cast<float*>(GetData())[index] = value;
        } else {
            const long quantized = std::lround(value / m_quant.scale) + m_quant.offset;
            if (m_type == TensorType::INT8) {
                static_cast<int8_t*>(GetData())[index] = std::clamp(quantized, -128L, 127L);
            } else {
                static_cast<uint8_t*>(GetData())[index] = std::clamp(quantized, 0L, 255L);
            }
        }
    }

private:
    std::vector<size_t> m_shape;
    TensorType m_type;
    TensorLayout m_layout;
    QuantParams m_quant;
    std::vector<uint8_t> m_data;
};

#define CHECK(condition)                                                         \
    do {                                                                         \
        if (!(condition)) {                                                      \
            std::fprintf(stderr, "%s:%d: %s\n", __func__, __LINE__, #condition); \
            return false;                                                        \
        }                                                                        \
    } while (false)

bool TestAngleDoesNotChangeConfidence(size_t bins, TensorType type, TensorLayout layout)
{
    std::vector<std::shared_ptr<TensorIface>> outputs;
    const size_t channels = 4 * bins + 15;
    for (size_t side : {size_t{4}, size_t{2}, size_t{1}}) {
        auto head = std::make_shared<MemoryTensor>(
            layout == TensorLayout::NCHW ? std::vector<size_t>{1, channels, side, side}
                                         : std::vector<size_t>{1, side, side, channels},
            type,
            layout,
            QuantParams{0.125F, type == TensorType::UINT8 ? 128 : 0});
        for (size_t index = 0; index < head->GetNumElements(); ++index) {
            head->Set(index, -16.0F);
        }
        for (size_t cell = 0; cell < side * side; ++cell) {
            for (size_t edge = 0; edge < 4; ++edge) {
                const size_t channel = edge * bins + 1;
                head->Set(layout == TensorLayout::NCHW ? channel * side * side + cell
                                                       : cell * channels + channel,
                          15.0F);
            }
        }
        if (side == 4) {
            const size_t classChannel = 4 * bins + 14;
            head->Set(layout == TensorLayout::NCHW ? classChannel * 16 + 5
                                                   : 5 * channels + classChannel,
                      2.0F);
        }
        outputs.push_back(head);
    }
    auto angle =
        std::make_shared<MemoryTensor>(std::vector<size_t>{1, 1, 21},
                                       type,
                                       layout,
                                       QuantParams{0.0625F, type == TensorType::UINT8 ? 128 : 0});
    outputs.push_back(angle);
    YoloV8PostProcessParams params{32, 32, 15, 20, 0.25F, 0.45F};
    std::vector<yolov8_detection::DetectionResult> results;
    YoloV8NewPostProcess process{outputs, results, params};
    for (float logit : {-4.0F, 0.0F, 4.0F}) {
        for (size_t index = 0; index < 21; ++index) {
            angle->Set(index, logit);
        }
        CHECK(process.DoPostProcess());
        CHECK(results.size() == 1);
        CHECK(results[0].classId == 14);
        CHECK(std::abs(results[0].score - 0.880797078F) < 1e-6F);
        CHECK(std::abs(results[0].angleRadians - (1.0F / (1.0F + std::exp(-logit)) - 0.25F) *
                                                     3.14159265358979323846F) < 1e-6F);
    }
    params.angleIsLogits = false;
    YoloV8NewPostProcess radiansProcess{outputs, results, params};
    CHECK(radiansProcess.DoPostProcess());
    CHECK(results.size() == 1 && results[0].angleRadians == 4.0F);

    outputs.pop_back();
    YoloV8NewPostProcess noAngleProcess{outputs, results, params};
    CHECK(noAngleProcess.DoPostProcess());
    CHECK(results.size() == 1 && results[0].angleRadians == 0.0F);
    CHECK(results[0].classId == 14);
    CHECK(std::abs(results[0].score - 0.880797078F) < 1e-6F);

    params.numClasses = 14;
    YoloV8NewPostProcess wrongClassCount{outputs, results, params};
    CHECK(!wrongClassCount.DoPostProcess());
    return true;
}

bool TestRgbPreprocessing(TensorType type, TensorLayout layout)
{
    auto input = std::make_shared<MemoryTensor>(
        layout == TensorLayout::NCHW ? std::vector<size_t>{1, 3, 2, 4}
                                     : std::vector<size_t>{1, 2, 4, 3},
        type,
        layout,
        QuantParams{1.0F / 255.0F, type == TensorType::INT8 ? -128 : 0});
    const uint8_t rgb[] = {255, 0, 0, 0,  255, 0,   0,  0,  255, 255, 255, 255,
                           0,   0, 0, 24, 80,  111, 15, 79, 165, 127, 128, 129};
    YoloV8PreProcess process{input};
    CHECK(process.DoPreProcess(rgb, sizeof(rgb)));
    for (size_t pixel = 0; pixel < 8; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            const size_t index =
                layout == TensorLayout::NCHW ? channel * 8 + pixel : pixel * 3 + channel;
            const uint8_t expected = rgb[pixel * 3 + channel];
            if (type == TensorType::FP32) {
                CHECK(std::abs(static_cast<float*>(input->GetData())[index] - expected / 255.0F) <
                      1e-6F);
            } else if (type == TensorType::INT8) {
                CHECK(static_cast<int8_t*>(input->GetData())[index] == expected - 128);
            } else {
                CHECK(static_cast<uint8_t*>(input->GetData())[index] == expected);
            }
        }
    }
    return true;
}

bool TestGrayscalePreprocessing(TensorType type, TensorLayout layout)
{
    auto input = std::make_shared<MemoryTensor>(
        layout == TensorLayout::NCHW ? std::vector<size_t>{1, 1, 2, 2}
                                     : std::vector<size_t>{1, 2, 2, 1},
        type,
        layout,
        QuantParams{1.0F / 255.0F, type == TensorType::INT8 ? -128 : 0});
    const uint8_t rgb[]  = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255};
    const int expected[] = {76, 150, 29, 255};
    YoloV8PreProcess process{input};
    CHECK(process.DoPreProcess(rgb, sizeof(rgb)));
    for (size_t index = 0; index < 4; ++index) {
        if (type == TensorType::FP32) {
            CHECK(std::abs(static_cast<float*>(input->GetData())[index] -
                           expected[index] / 255.0F) < 1e-6F);
        } else if (type == TensorType::INT8) {
            CHECK(static_cast<int8_t*>(input->GetData())[index] == expected[index] - 128);
        } else {
            CHECK(static_cast<uint8_t*>(input->GetData())[index] == expected[index]);
        }
    }
    CHECK(!process.DoPreProcess(rgb, 3));
    return true;
}

bool TestOrdinaryBox()
{
    auto tensor = std::make_shared<MemoryTensor>(
        std::vector<size_t>{1, 5, 2}, TensorType::FP32, TensorLayout::NCHW);
    for (size_t box = 0; box < 2; ++box) {
        tensor->Set(box, 16);
        tensor->Set(2 + box, 16);
        tensor->Set(4 + box, 20);
        tensor->Set(6 + box, 8);
        tensor->Set(8 + box, box == 0 ? 0.9F : 0.8F);
    }
    YoloV8PostProcessParams params{32, 32, 1, 20, 0.25F, 0.45F};
    std::vector<yolov8_detection::DetectionResult> results;
    YoloV8PostProcess process{tensor, results, params};
    CHECK(process.DoPostProcess());
    CHECK(results.size() == 1);
    CHECK(results[0].classId == 0 && results[0].score == 0.9F);
    CHECK(results[0].centerX == 16 && results[0].centerY == 16);
    CHECK(results[0].width == 20 && results[0].height == 8);
    CHECK(results[0].angleRadians == 0);
    return true;
}

bool TestRotatedNms()
{
    std::vector<std::shared_ptr<TensorIface>> outputs;
    for (size_t side : {size_t{4}, size_t{2}, size_t{1}}) {
        auto head = std::make_shared<MemoryTensor>(
            std::vector<size_t>{1, 79, side, side}, TensorType::FP32, TensorLayout::NCHW);
        for (size_t index = 0; index < head->GetNumElements(); ++index) {
            head->Set(index, -100);
        }
        if (side == 4) {
            for (size_t cell : {size_t{5}, size_t{6}}) {
                for (size_t edge : {size_t{0}, size_t{2}}) {
                    head->Set((edge * 16 + 2) * 16 + cell, 100);
                }
                for (size_t edge : {size_t{1}, size_t{3}}) {
                    head->Set(edge * 16 * 16 + cell, 100);
                    head->Set((edge * 16 + 1) * 16 + cell, 100);
                }
                head->Set(64 * 16 + cell, 2);
            }
        }
        outputs.push_back(head);
    }
    auto angle = std::make_shared<MemoryTensor>(
        std::vector<size_t>{1, 1, 21}, TensorType::FP32, TensorLayout::NCHW);
    angle->Set(5, -100);
    angle->Set(6, 0);
    outputs.push_back(angle);
    YoloV8PostProcessParams params{32, 32, 15, 20, 0.25F, 0.45F};
    std::vector<yolov8_detection::DetectionResult> results;
    YoloV8NewPostProcess process{outputs, results, params};
    CHECK(process.DoPostProcess());
    CHECK(results.size() == 2);
    return true;
}

bool TestQrCodeInputLookup(TensorType type, size_t width, size_t height)
{
    auto input = std::make_shared<MemoryTensor>(
        std::vector<size_t>{1, height, width, 1}, type, TensorLayout::NHWC);
    QrCodePreProcess process{input};
    std::vector<uint8_t> image(width * height);
    for (size_t index = 0; index < image.size(); ++index) {
        image[index] = static_cast<uint8_t>(index);
    }
    for (QuantParams quant : {QuantParams{0.0037831603549420834F, -128},
                              QuantParams{0.01F, 17},
                              QuantParams{1.0F / 255.0F, -128},
                              QuantParams{0.0037831603549420834F, 0},
                              QuantParams{0.02F, 119}}) {
        input->SetQuantParams(quant);
        for (size_t repeat = 0; repeat < 2; ++repeat) {
            CHECK(process.DoPreProcess(image.data(), image.size()));
            for (size_t index = 0; index < image.size(); ++index) {
                if (type == TensorType::FP32) {
                    CHECK(static_cast<float*>(input->GetData())[index] == image[index] / 255.0F);
                } else {
                    const long expected =
                        std::lround((image[index] / 255.0F) / quant.scale) + quant.offset;
                    if (type == TensorType::INT8) {
                        CHECK(static_cast<int8_t*>(input->GetData())[index] ==
                              std::clamp(expected, -128L, 127L));
                    } else {
                        CHECK(static_cast<uint8_t*>(input->GetData())[index] ==
                              std::clamp(expected, 0L, 255L));
                    }
                }
            }
            std::reverse(image.begin(), image.end());
        }
    }
    CHECK(!process.DoPreProcess(nullptr, image.size()));
    CHECK(!process.DoPreProcess(image.data(), image.size() - 1));
    if (type != TensorType::FP32) {
        input->SetQuantParams({0.0F, 0});
        CHECK(!process.DoPreProcess(image.data(), image.size()));
    }
    return true;
}

bool TestRectangularQrCodeHeads(TensorLayout layout)
{
    constexpr size_t width    = 32;
    constexpr size_t height   = 64;
    constexpr size_t channels = 47;
    std::vector<std::shared_ptr<TensorIface>> outputs;
    for (size_t stride : {size_t{8}, size_t{16}, size_t{32}}) {
        const size_t rows    = height / stride;
        const size_t columns = width / stride;
        const size_t cells   = rows * columns;
        auto head            = std::make_shared<MemoryTensor>(
            layout == TensorLayout::NCHW ? std::vector<size_t>{1, channels, rows, columns}
                                         : std::vector<size_t>{1, rows, columns, channels},
            TensorType::FP32,
            layout);
        for (size_t index = 0; index < head->GetNumElements(); ++index) {
            head->Set(index, -100.0F);
        }
        if (stride == 8) {
            constexpr size_t cell = 2 * 4 + 1;
            for (size_t side = 0; side < 4; ++side) {
                const size_t channel = side * 8 + 1;
                head->Set(layout == TensorLayout::NCHW ? channel * cells + cell
                                                       : cell * channels + channel,
                          100.0F);
            }
            head->Set(layout == TensorLayout::NCHW ? 32 * cells + cell : cell * channels + 32,
                      4.0F);
        }
        outputs.push_back(head);
    }
    auto angle =
        std::make_shared<MemoryTensor>(std::vector<size_t>{1, 1, 42}, TensorType::FP32, layout);
    outputs.push_back(angle);
    const YoloV8PostProcessParams params{width, height, 15, 20, 0.25F, 0.45F};
    std::vector<yolov8_detection::DetectionResult> results;
    YoloV8NewPostProcess process{outputs, results, params};
    CHECK(process.DoPostProcess());
    CHECK(results.size() == 1);
    CHECK(results[0].classId == 0);
    CHECK(std::abs(results[0].centerX - 12.0F) < 1e-4F);
    CHECK(std::abs(results[0].centerY - 20.0F) < 1e-4F);
    CHECK(std::abs(results[0].width - 16.0F) < 1e-4F);
    CHECK(std::abs(results[0].height - 16.0F) < 1e-4F);
    return true;
}

bool TestQuantizedDflParity(size_t bins, TensorType type, TensorLayout layout)
{
    std::vector<std::shared_ptr<TensorIface>> integerOutputs;
    std::vector<std::shared_ptr<TensorIface>> floatOutputs;
    const size_t channels = 4 * bins + 15;
    size_t headIndex      = 0;
    for (size_t side : {size_t{4}, size_t{2}, size_t{1}, size_t{0}}) {
        const bool angle     = side == 0;
        const auto shape     = angle ? std::vector<size_t>{1, 1, 21}
                               : layout == TensorLayout::NCHW
                                   ? std::vector<size_t>{1, channels, side, side}
                                   : std::vector<size_t>{1, side, side, channels};
        const float scales[] = {0.037123F, 0.07191F, 0.02341F, 0.01357F};
        auto quantized       = std::make_shared<MemoryTensor>(
            shape,
            type,
            layout,
            QuantParams{scales[headIndex++], type == TensorType::INT8 ? -17 : 137});
        auto floating = std::make_shared<MemoryTensor>(shape, TensorType::FP32, layout);
        for (size_t index = 0; index < quantized->GetNumElements(); ++index) {
            quantized->Set(index, -8.0F);
        }
        if (angle) {
            for (size_t index = 0; index < 21; ++index) {
                quantized->Set(index, (static_cast<int>(index % 7) - 3) * 0.417F);
            }
        } else {
            for (size_t cell = 0; cell < side * side; ++cell) {
                for (size_t channel = 0; channel < 4 * bins; ++channel) {
                    const size_t index = layout == TensorLayout::NCHW ? channel * side * side + cell
                                                                      : cell * channels + channel;
                    quantized->Set(
                        index, (static_cast<int>((channel * 37 + cell * 19) % 211) - 105) * 0.043F);
                }
                if (cell % 3 == 0) {
                    const size_t channel = 4 * bins + cell % 15;
                    quantized->Set(layout == TensorLayout::NCHW ? channel * side * side + cell
                                                                : cell * channels + channel,
                                   1.823F);
                }
            }
        }
        for (size_t index = 0; index < quantized->GetNumElements(); ++index) {
            floating->Set(index, quantized->Value(index));
        }
        integerOutputs.push_back(quantized);
        floatOutputs.push_back(floating);
    }
    const YoloV8PostProcessParams params{32, 32, 15, 20, 0.25F, 0.45F};
    std::vector<yolov8_detection::DetectionResult> integerResults, floatResults;
    YoloV8NewPostProcess integerProcess{integerOutputs, integerResults, params};
    YoloV8NewPostProcess floatProcess{floatOutputs, floatResults, params};
    CHECK(floatProcess.DoPostProcess());
    CHECK(integerProcess.DecodeDetections());
    CHECK(integerResults.empty());
    integerProcess.SuppressDetections();
    for (size_t pass = 0; pass < 3; ++pass) {
        CHECK(!integerResults.empty() && integerResults.size() == floatResults.size());
        for (size_t index = 0; index < integerResults.size(); ++index) {
            const auto& integer  = integerResults[index];
            const auto& floating = floatResults[index];
            CHECK(integer.classId == floating.classId);
            CHECK(integer.score == floating.score);
            CHECK(integer.angleRadians == floating.angleRadians);
            CHECK(std::abs(integer.centerX - floating.centerX) < 2e-4F);
            CHECK(std::abs(integer.centerY - floating.centerY) < 2e-4F);
            CHECK(std::abs(integer.width - floating.width) < 2e-4F);
            CHECK(std::abs(integer.height - floating.height) < 2e-4F);
        }
        if (pass == 1) {
            // Rebind the same processor to changed scales/zero points after a warm call.
            for (size_t index = 0; index < integerOutputs.size(); ++index) {
                auto quantized = std::static_pointer_cast<MemoryTensor>(integerOutputs[index]);
                auto floating  = std::static_pointer_cast<MemoryTensor>(floatOutputs[index]);
                auto quant     = quantized->GetQuantParams();
                quant.scale *= 0.93F;
                quant.offset += 3;
                quantized->SetQuantParams(quant);
                for (size_t value = 0; value < quantized->GetNumElements(); ++value) {
                    floating->Set(value, quantized->Value(value));
                }
            }
        }
        CHECK(integerProcess.DoPostProcess());
        CHECK(floatProcess.DoPostProcess());
    }
    return true;
}

int DecodeFile(const char* path,
               size_t imageSize,
               size_t channels,
               uint32_t numClasses,
               TensorType type = TensorType::FP32)
{
    std::ifstream stream{path, std::ios::binary};
    // Integer fixture format: scale (float32), zero point (int32), then raw bytes per tensor.
    const auto readTensor = [&stream, type](const std::shared_ptr<MemoryTensor>& tensor) {
        if (type != TensorType::FP32) {
            float scale    = 0.0F;
            int32_t offset = 0;
            stream.read(reinterpret_cast<char*>(&scale), sizeof(scale));
            stream.read(reinterpret_cast<char*>(&offset), sizeof(offset));
            tensor->SetQuantParams({scale, offset});
        }
        stream.read(static_cast<char*>(tensor->GetData()), tensor->Bytes());
    };
    std::vector<std::shared_ptr<TensorIface>> outputs;
    for (size_t stride : {size_t{8}, size_t{16}, size_t{32}}) {
        const size_t size = imageSize / stride;
        auto tensor       = std::make_shared<MemoryTensor>(
            std::vector<size_t>{1, channels, size, size}, type, TensorLayout::NCHW);
        readTensor(tensor);
        outputs.push_back(tensor);
    }
    const size_t anchors = (imageSize / 8) * (imageSize / 8) + (imageSize / 16) * (imageSize / 16) +
                           (imageSize / 32) * (imageSize / 32);
    auto angle           = std::make_shared<MemoryTensor>(
        std::vector<size_t>{1, 1, anchors}, type, TensorLayout::NCHW);
    readTensor(angle);
    if (!stream) {
        return 1;
    }
    outputs.push_back(angle);
    YoloV8PostProcessParams params{static_cast<uint32_t>(imageSize),
                                   static_cast<uint32_t>(imageSize),
                                   numClasses,
                                   20,
                                   0.25F,
                                   0.45F};
    std::vector<yolov8_detection::DetectionResult> results;
    YoloV8NewPostProcess process{outputs, results, params};
    if (!process.DoPostProcess()) {
        return 1;
    }
    for (const auto& result : results) {
        std::printf("%d %.9f %.9f %.9f %.9f %.9f %.9f\n",
                    result.classId,
                    result.score,
                    result.centerX,
                    result.centerY,
                    result.width,
                    result.height,
                    result.angleRadians);
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 5 || argc == 6) {
        TensorType type = TensorType::FP32;
        if (argc == 6) {
            const std::string name{argv[5]};
            if (name != "int8" && name != "uint8") {
                return 1;
            }
            type = name == "int8" ? TensorType::INT8 : TensorType::UINT8;
        }
        return DecodeFile(
            argv[1], std::stoul(argv[2]), std::stoul(argv[3]), std::stoul(argv[4]), type);
    }
    if (!TestRotatedNms() || !TestOrdinaryBox()) {
        return 1;
    }
    for (TensorLayout layout : {TensorLayout::NCHW, TensorLayout::NHWC}) {
        if (!TestRectangularQrCodeHeads(layout)) {
            return 1;
        }
    }
    for (TensorType type : {TensorType::FP32, TensorType::INT8, TensorType::UINT8}) {
        if (!TestQrCodeInputLookup(type, 256, 256) ||
            (type == TensorType::INT8 && !TestQrCodeInputLookup(type, 320, 192))) {
            return 1;
        }
        for (TensorLayout layout : {TensorLayout::NCHW, TensorLayout::NHWC}) {
            for (size_t bins : {size_t{16}, size_t{8}}) {
                if (!TestAngleDoesNotChangeConfidence(bins, type, layout) ||
                    (type != TensorType::FP32 && !TestQuantizedDflParity(bins, type, layout))) {
                    return 1;
                }
            }
            if (!TestGrayscalePreprocessing(type, layout) || !TestRgbPreprocessing(type, layout)) {
                return 1;
            }
        }
    }
    std::puts("YOLOv8 confidence, class and grayscale processing tests passed");
    return 0;
}
