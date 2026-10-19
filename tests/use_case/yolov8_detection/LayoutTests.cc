/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/yolov8_detection/YoloV8PostProcessing.hpp"
#include "mlek/use_case/yolov8_detection/YoloV8PreProcessing.hpp"

#include <catch.hpp>
#include <utility>

namespace {

class TestTensor : public arm::app::fwk::iface::TensorIface {
public:
    TestTensor(std::vector<size_t> shape,
               arm::app::fwk::iface::TensorLayout layout,
               std::vector<float> data) :
        m_shape(std::move(shape)), m_layout(layout), m_data(std::move(data))
    {}

    void* GetData() override
    { return m_data.data(); }
    size_t Bytes() override
    { return m_data.size() * sizeof(float); }
    size_t GetNumElements() override
    { return m_data.size(); }
    arm::app::fwk::iface::TensorType Type() override
    { return arm::app::fwk::iface::TensorType::FP32; }
    arm::app::fwk::iface::TensorLayout Layout() override
    { return m_layout; }
    std::vector<size_t> Shape() override
    { return m_shape; }
    arm::app::fwk::iface::QuantParams GetQuantParams() override
    { return {}; }

private:
    std::vector<size_t> m_shape;
    arm::app::fwk::iface::TensorLayout m_layout;
    std::vector<float> m_data;
};

} // namespace

TEST_CASE("YOLO input honors channels-last storage with NCHW logical sizes", "[yolov8]")
{
    using arm::app::fwk::iface::TensorLayout;
    const uint8_t pixels[] = {0, 32, 64, 96, 128, 160};
    for (const auto layout : {TensorLayout::NCHW, TensorLayout::NHWC}) {
        auto tensor = std::make_shared<TestTensor>(
            std::vector<size_t>{1, 3, 1, 2}, layout, std::vector<float>(6));
        arm::app::YoloV8PreProcess process(tensor, true);
        REQUIRE(process.DoPreProcess(pixels, sizeof(pixels)));
        auto* actual = static_cast<float*>(tensor->GetData());
        for (size_t pixel = 0; pixel < 2; ++pixel) {
            for (size_t channel = 0; channel < 3; ++channel) {
                const size_t index =
                    layout == TensorLayout::NHWC ? pixel * 3 + channel : channel * 2 + pixel;
                CHECK(actual[index] == Approx(pixels[pixel * 3 + channel] / 255.0F));
            }
        }
    }
}

TEST_CASE("YOLO raw heads produce the same detections for NCHW and NHWC storage", "[yolov8]")
{
    using arm::app::fwk::iface::TensorIface;
    using arm::app::fwk::iface::TensorLayout;
    using arm::app::yolov8_detection::DetectionResult;
    std::vector<DetectionResult> reference;
    const arm::app::YoloV8PostProcessParams params{32, 32, 1, 20, 0.25F, 0.45F};
    for (const auto layout : {TensorLayout::NCHW, TensorLayout::NHWC}) {
        std::vector<std::shared_ptr<TensorIface>> outputs;
        for (const size_t size : {4, 2, 1}) {
            const size_t cells = size * size;
            std::vector<float> data(65 * cells);
            for (size_t channel = 0; channel < 65; ++channel) {
                for (size_t cell = 0; cell < cells; ++cell) {
                    const float value = channel == 64 ? (size == 4 && cell == 5 ? 8.0F : -8.0F)
                                                      : (channel % 16 == 1 ? 8.0F : -8.0F);
                    const size_t index =
                        layout == TensorLayout::NHWC ? cell * 65 + channel : channel * cells + cell;
                    data[index] = value;
                }
            }
            outputs.push_back(std::make_shared<TestTensor>(
                std::vector<size_t>{1, 65, size, size}, layout, std::move(data)));
        }
        outputs.push_back(std::make_shared<TestTensor>(
            std::vector<size_t>{1, 1, 21}, TensorLayout::INVALID, std::vector<float>(21, 8.0F)));
        std::vector<DetectionResult> actual;
        arm::app::YoloV8NewPostProcess process(outputs, actual, params, true);
        REQUIRE(process.DoPostProcess());
        REQUIRE(actual.size() == 1);
        if (layout == TensorLayout::NCHW) {
            reference = actual;
        } else {
            CHECK(actual[0].score == Approx(reference[0].score));
            CHECK(actual[0].classId == reference[0].classId);
            CHECK(actual[0].centerX == Approx(reference[0].centerX));
            CHECK(actual[0].centerY == Approx(reference[0].centerY));
            CHECK(actual[0].width == reference[0].width);
            CHECK(actual[0].height == reference[0].height);
        }
    }
}

TEST_CASE("YOLO TFLite input infers storage from shape", "[yolov8]")
{
    using arm::app::fwk::iface::TensorLayout;
    const uint8_t pixels[] = {0, 32, 64, 96, 128, 160};
    auto tensor            = std::make_shared<TestTensor>(
        std::vector<size_t>{1, 3, 1, 2}, TensorLayout::NHWC, std::vector<float>(6));
    arm::app::YoloV8PreProcess process(tensor);
    REQUIRE(process.DoPreProcess(pixels, sizeof(pixels)));
    auto* actual = static_cast<float*>(tensor->GetData());
    for (size_t pixel = 0; pixel < 2; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            CHECK(actual[channel * 2 + pixel] == Approx(pixels[pixel * 3 + channel] / 255.0F));
        }
    }
}

TEST_CASE("YOLO native NHWC input can use shape inference", "[yolov8]")
{
    using arm::app::fwk::iface::TensorLayout;
    const uint8_t pixels[] = {0, 32, 64, 96, 128, 160};
    auto tensor            = std::make_shared<TestTensor>(
        std::vector<size_t>{1, 1, 2, 3}, TensorLayout::NCHW, std::vector<float>(6));
    arm::app::YoloV8PreProcess process(tensor, false);
    REQUIRE(process.DoPreProcess(pixels, sizeof(pixels)));
    auto* actual = static_cast<float*>(tensor->GetData());
    for (size_t index = 0; index < 6; ++index) {
        CHECK(actual[index] == Approx(pixels[index] / 255.0F));
    }
}

TEST_CASE("YOLO OBB accepts singleton four dimensional angle outputs", "[yolov8]")
{
    using arm::app::fwk::iface::TensorIface;
    using arm::app::fwk::iface::TensorLayout;
    using arm::app::yolov8_detection::DetectionResult;
    const arm::app::YoloV8PostProcessParams params{32, 32, 1, 1, 0.25F, 0.45F, 8, true};
    std::vector<DetectionResult> reference;
    for (const auto& angleShape : {std::vector<size_t>{1, 1, 21},
                                   std::vector<size_t>{1, 1, 21, 1},
                                   std::vector<size_t>{1, 21, 1, 1}}) {
        std::vector<std::shared_ptr<TensorIface>> outputs;
        for (const size_t size : {4, 2, 1}) {
            std::vector<float> data(33 * size * size, -8.0F);
            for (size_t cell = 0; cell < size * size; ++cell) {
                for (size_t side = 0; side < 4; ++side) {
                    data[cell * 33 + side * 8 + 1] = 8.0F;
                }
            }
            if (size == 4) {
                data[5 * 33 + 32] = 8.0F;
            }
            outputs.push_back(std::make_shared<TestTensor>(
                std::vector<size_t>{1, size, size, 33}, TensorLayout::NCHW, std::move(data)));
        }
        outputs.push_back(std::make_shared<TestTensor>(
            angleShape, TensorLayout::NCHW, std::vector<float>(21, 1.0F)));
        std::vector<DetectionResult> actual;
        arm::app::YoloV8NewPostProcess process(outputs, actual, params, false);
        REQUIRE(process.DoPostProcess());
        REQUIRE(actual.size() == 1);
        if (reference.empty()) {
            reference = actual;
        }
        CHECK(actual[0].score == Approx(reference[0].score));
        CHECK(actual[0].angleRadians == Approx(reference[0].angleRadians));
        CHECK(actual[0].centerX == Approx(reference[0].centerX));
        CHECK(actual[0].centerY == Approx(reference[0].centerY));
        CHECK(actual[0].width == reference[0].width);
        CHECK(actual[0].height == reference[0].height);
    }
}
