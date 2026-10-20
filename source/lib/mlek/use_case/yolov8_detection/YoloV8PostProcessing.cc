/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/yolov8_detection/YoloV8PostProcessing.hpp"

#include "mlek/log/log_macros.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace arm::app {

namespace {

    float CalculateRotatedIoU(const yolov8_detection::DetectionResult& first,
                              const yolov8_detection::DetectionResult& second)
    {
        const auto covariance = [](const auto& box) {
            const float widthVariance  = box.width * box.width / 12.0F;
            const float heightVariance = box.height * box.height / 12.0F;
            const float cosine         = std::cos(box.angleRadians);
            const float sine           = std::sin(box.angleRadians);
            return std::array<float, 3>{
                widthVariance * cosine * cosine + heightVariance * sine * sine,
                widthVariance * sine * sine + heightVariance * cosine * cosine,
                (widthVariance - heightVariance) * cosine * sine};
        };
        const auto firstCov      = covariance(first);
        const auto secondCov     = covariance(second);
        const float a            = firstCov[0] + secondCov[0];
        const float b            = firstCov[1] + secondCov[1];
        const float c            = firstCov[2] + secondCov[2];
        const float dx           = first.centerX - second.centerX;
        const float dy           = first.centerY - second.centerY;
        constexpr float kEpsilon = 1e-7F;
        const float determinant  = a * b - c * c;
        const float firstDet =
            std::max(0.0F, firstCov[0] * firstCov[1] - firstCov[2] * firstCov[2]);
        const float secondDet =
            std::max(0.0F, secondCov[0] * secondCov[1] - secondCov[2] * secondCov[2]);
        const float translation =
            (0.25F * (a * dy * dy + b * dx * dx) - 0.5F * c * dx * dy) / (determinant + kEpsilon);
        const float shape =
            0.5F *
            std::log(determinant / (4.0F * std::sqrt(firstDet * secondDet) + kEpsilon) + kEpsilon);
        const float distance = std::clamp(translation + shape, kEpsilon, 100.0F);
        return std::clamp(1.0F - std::sqrt(1.0F - std::exp(-distance) + kEpsilon), 0.0F, 1.0F);
    }

    float CalculateDetectionIoU(const yolov8_detection::DetectionResult& first,
                                const yolov8_detection::DetectionResult& second)
    {
        const float intersectionLeft =
            std::max(first.centerX - first.width * 0.5F, second.centerX - second.width * 0.5F);
        const float intersectionTop =
            std::max(first.centerY - first.height * 0.5F, second.centerY - second.height * 0.5F);
        const float intersectionRight =
            std::min(first.centerX + first.width * 0.5F, second.centerX + second.width * 0.5F);
        const float intersectionBottom =
            std::min(first.centerY + first.height * 0.5F, second.centerY + second.height * 0.5F);

        const float intersectionWidth  = std::max(0.0F, intersectionRight - intersectionLeft);
        const float intersectionHeight = std::max(0.0F, intersectionBottom - intersectionTop);

        const float intersectionArea = intersectionWidth * intersectionHeight;
        const float firstArea        = first.width * first.height;
        const float secondArea       = second.width * second.height;
        const float unionArea        = firstArea + secondArea - intersectionArea;

        return unionArea > 0.0F ? intersectionArea / unionArea : 0.0F;
    }

    void ApplyClassAwareNms(std::vector<yolov8_detection::DetectionResult>& candidates,
                            std::vector<yolov8_detection::DetectionResult>& results,
                            const YoloV8PostProcessParams& params,
                            bool rotatedNms = false)
    {
        if (params.maxDetections == 0) {
            return;
        }

        std::sort(candidates.begin(), candidates.end(), [](const auto& first, const auto& second) {
            return first.score > second.score;
        });

        for (const auto& candidate : candidates) {
            bool suppressed = false;

            for (const auto& selected : results) {
                if (candidate.classId != selected.classId) {
                    continue;
                }

                const float overlap = rotatedNms ? CalculateRotatedIoU(candidate, selected)
                                                 : CalculateDetectionIoU(candidate, selected);
                if (overlap > params.nmsThreshold) {
                    suppressed = true;
                    break;
                }
            }

            if (!suppressed) {
                results.push_back(candidate);

                if (results.size() >= params.maxDetections) {
                    break;
                }
            }
        }
    }

    float Dequantize(const int8_t value, const fwk::iface::QuantParams& quantParams)
    {
        return (static_cast<float>(value) - static_cast<float>(quantParams.offset)) *
               quantParams.scale;
    }

    float Sigmoid(const float value)
    { return 1.0F / (1.0F + std::exp(-value)); }

    /** Cache virtual tensor access once, rather than once per output element. */
    struct OutputReader {
        const void* data{nullptr};
        size_t elements{0};
        fwk::iface::TensorType type{fwk::iface::TensorType::INVALID};
        fwk::iface::QuantParams quant;

        bool Init(const std::shared_ptr<fwk::iface::TensorIface>& tensor)
        {
            data     = tensor->GetData();
            elements = tensor->GetNumElements();
            type     = tensor->Type();
            if (type == fwk::iface::TensorType::INT8 || type == fwk::iface::TensorType::UINT8) {
                quant = tensor->GetQuantParams();
                return data != nullptr && std::isfinite(quant.scale) && quant.scale > 0.0F;
            }
            return data != nullptr && type == fwk::iface::TensorType::FP32;
        }

        bool IsQuantized() const
        { return type != fwk::iface::TensorType::FP32; }

        bool ReadRaw(size_t index, float& value) const
        {
            if (index >= elements) {
                return false;
            }
            if (type == fwk::iface::TensorType::INT8) {
                value = static_cast<const int8_t*>(data)[index];
            } else if (type == fwk::iface::TensorType::UINT8) {
                value = static_cast<const uint8_t*>(data)[index];
            } else {
                value = static_cast<const float*>(data)[index];
            }
            return std::isfinite(value);
        }

        float ToFloat(float value) const
        { return IsQuantized() ? (value - quant.offset) * quant.scale : value; }

        size_t LookupIndex(float value) const
        {
            return static_cast<size_t>(static_cast<int>(value) +
                                       (type == fwk::iface::TensorType::INT8 ? 128 : 0));
        }
    };

    float ToProbability(const float value)
    {
        /* Ultralytics exports probabilities, while some quantized heads expose logits. */
        if (value >= -0.01F && value <= 1.01F) {
            return std::clamp(value, 0.0F, 1.0F);
        }
        return Sigmoid(value);
    }

    bool ReadTensorValue(const std::shared_ptr<fwk::iface::TensorIface>& tensor,
                         size_t index,
                         float& value)
    {
        if (tensor == nullptr || index >= tensor->GetNumElements()) {
            return false;
        }

        switch (tensor->Type()) {
        case fwk::iface::TensorType::FP32:
            value = tensor->GetData<float>()[index];
            return std::isfinite(value);
        case fwk::iface::TensorType::INT8: {
            const auto quantParams = tensor->GetQuantParams();
            if (!std::isfinite(quantParams.scale) || quantParams.scale <= 0.0F) {
                return false;
            }
            value = Dequantize(tensor->GetData<int8_t>()[index], quantParams);
            return true;
        }
        case fwk::iface::TensorType::UINT8: {
            const auto quantParams = tensor->GetQuantParams();
            if (!std::isfinite(quantParams.scale) || quantParams.scale <= 0.0F) {
                return false;
            }
            value = (static_cast<float>(tensor->GetData<uint8_t>()[index]) -
                     static_cast<float>(quantParams.offset)) *
                    quantParams.scale;
            return true;
        }
        default:
            return false;
        }
    }

    enum class SingleOutputLayout { AttributesFirst, BoxesFirst };

    struct SingleOutputDescription {
        size_t attributes{0};
        size_t boxCount{0};
        SingleOutputLayout layout{SingleOutputLayout::AttributesFirst};
    };

    bool DescribeSingleOutput(const std::vector<size_t>& shape,
                              uint32_t numClasses,
                              SingleOutputDescription& description)
    {
        size_t firstDimension  = 0;
        size_t secondDimension = 0;

        if (shape.size() == 3 && shape[0] == 1) {
            firstDimension  = shape[1];
            secondDimension = shape[2];
        } else if (shape.size() == 2) {
            firstDimension  = shape[0];
            secondDimension = shape[1];
        } else {
            return false;
        }

        const size_t classOnlyAttributes  = 4 + numClasses;
        const size_t objectnessAttributes = 5 + numClasses;
        if (firstDimension == classOnlyAttributes || firstDimension == objectnessAttributes) {
            description.attributes = firstDimension;
            description.boxCount   = secondDimension;
            description.layout     = SingleOutputLayout::AttributesFirst;
        } else if (secondDimension == classOnlyAttributes ||
                   secondDimension == objectnessAttributes) {
            description.attributes = secondDimension;
            description.boxCount   = firstDimension;
            description.layout     = SingleOutputLayout::BoxesFirst;
        } else {
            return false;
        }

        return description.boxCount > 0;
    }

    size_t
    GetSingleOutputIndex(const SingleOutputDescription& description, size_t box, size_t attribute)
    {
        if (description.layout == SingleOutputLayout::AttributesFirst) {
            return attribute * description.boxCount + box;
        }
        return box * description.attributes + attribute;
    }

    bool IntersectsImage(const yolov8_detection::DetectionResult& result,
                         const YoloV8PostProcessParams& params)
    {
        if (!std::isfinite(result.centerX) || !std::isfinite(result.centerY) ||
            !std::isfinite(result.width) || !std::isfinite(result.height) ||
            !std::isfinite(result.angleRadians) || result.width <= 0.0F || result.height <= 0.0F) {
            return false;
        }
        const float cosine  = std::abs(std::cos(result.angleRadians));
        const float sine    = std::abs(std::sin(result.angleRadians));
        const float extentX = (cosine * result.width + sine * result.height) * 0.5F;
        const float extentY = (sine * result.width + cosine * result.height) * 0.5F;
        return result.centerX + extentX > 0.0F && result.centerY + extentY > 0.0F &&
               result.centerX - extentX < params.imageWidth &&
               result.centerY - extentY < params.imageHeight;
    }

    bool GetDetectionBox(const float centerX,
                         const float centerY,
                         const float boxWidth,
                         const float boxHeight,
                         const YoloV8PostProcessParams& params,
                         const bool normalized,
                         yolov8_detection::DetectionResult& result)
    {
        if (!std::isfinite(centerX) || !std::isfinite(centerY) || !std::isfinite(boxWidth) ||
            !std::isfinite(boxHeight)) {
            return false;
        }

        result.centerX = normalized ? centerX * static_cast<float>(params.imageWidth) : centerX;
        result.centerY = normalized ? centerY * static_cast<float>(params.imageHeight) : centerY;
        result.width   = normalized ? boxWidth * static_cast<float>(params.imageWidth) : boxWidth;
        result.height = normalized ? boxHeight * static_cast<float>(params.imageHeight) : boxHeight;

        result.angleRadians = 0.0F;
        return IntersectsImage(result, params);
    }

} /* anonymous namespace */

YoloV8PostProcess::YoloV8PostProcess(const std::shared_ptr<fwk::iface::TensorIface>& outputTensor,
                                     std::vector<yolov8_detection::DetectionResult>& results,
                                     const YoloV8PostProcessParams& params) :
    m_outputTensor{outputTensor}, m_results{results}, m_params{params}
{}

bool YoloV8PostProcess::DoPostProcess()
{
    m_results.clear();

    if (m_outputTensor == nullptr) {
        printf_err("YOLOv8 output tensor is unavailable\n");
        return false;
    }

    SingleOutputDescription description;
    if (!DescribeSingleOutput(m_outputTensor->Shape(), m_params.numClasses, description)) {
        printf_err("Expected YOLOv8 output shape [1, 4 + classes, boxes] or "
                   "[1, boxes, 4 + classes]\n");
        return false;
    }

    const bool hasObjectness = description.attributes == 5 + m_params.numClasses;
    float maximumCoordinate  = 0.0F;
    for (size_t box = 0; box < description.boxCount; ++box) {
        for (size_t coordinate = 0; coordinate < 4; ++coordinate) {
            float value = 0.0F;
            if (!ReadTensorValue(
                    m_outputTensor, GetSingleOutputIndex(description, box, coordinate), value)) {
                printf_err("YOLOv8 output type or quantization is unsupported\n");
                return false;
            }
            maximumCoordinate = std::max(maximumCoordinate, std::fabs(value));
        }
    }

    /* Normalized exports stay close to [0, 1]; pixel-space exports are much larger. */
    const bool normalizedCoordinates = maximumCoordinate <= 2.0F;
    std::vector<yolov8_detection::DetectionResult> candidates;
    candidates.reserve(
        std::min(description.boxCount, static_cast<size_t>(m_params.maxDetections) * 4U));

    for (size_t box = 0; box < description.boxCount; ++box) {
        float bestScore = 0.0F;
        int bestClass   = -1;

        float objectness = 1.0F;
        if (hasObjectness) {
            float objectnessValue = 0.0F;
            if (!ReadTensorValue(
                    m_outputTensor, GetSingleOutputIndex(description, box, 4), objectnessValue)) {
                printf_err("YOLOv8 objectness output is unavailable\n");
                return false;
            }
            objectness = ToProbability(objectnessValue);
        }

        for (size_t classIndex = 0; classIndex < m_params.numClasses; ++classIndex) {
            float classValue = 0.0F;
            if (!ReadTensorValue(m_outputTensor,
                                 GetSingleOutputIndex(
                                     description, box, 4 + classIndex + (hasObjectness ? 1 : 0)),
                                 classValue)) {
                printf_err("YOLOv8 class output is unavailable\n");
                return false;
            }
            const float score = objectness * ToProbability(classValue);

            if (score > bestScore) {
                bestScore = score;
                bestClass = static_cast<int>(classIndex);
            }
        }

        if (bestScore < m_params.scoreThreshold) {
            continue;
        }

        float centerX   = 0.0F;
        float centerY   = 0.0F;
        float boxWidth  = 0.0F;
        float boxHeight = 0.0F;
        if (!ReadTensorValue(m_outputTensor, GetSingleOutputIndex(description, box, 0), centerX) ||
            !ReadTensorValue(m_outputTensor, GetSingleOutputIndex(description, box, 1), centerY) ||
            !ReadTensorValue(m_outputTensor, GetSingleOutputIndex(description, box, 2), boxWidth) ||
            !ReadTensorValue(
                m_outputTensor, GetSingleOutputIndex(description, box, 3), boxHeight)) {
            printf_err("YOLOv8 box output is unavailable\n");
            continue;
        }

        yolov8_detection::DetectionResult result{bestScore, bestClass};
        if (GetDetectionBox(
                centerX, centerY, boxWidth, boxHeight, m_params, normalizedCoordinates, result)) {
            candidates.push_back(result);
        }
    }

    /* Class-aware NMS: different classes do not suppress each other. */
    ApplyClassAwareNms(candidates, m_results, m_params);

    return true;
}

void YoloV8NewPostProcess::LookupTables::Update(const fwk::iface::QuantParams& params,
                                                fwk::iface::TensorType tensorType,
                                                bool dfl)
{
    if (type == tensorType && quant.scale == params.scale && quant.offset == params.offset) {
        return;
    }
    quant = params;
    type  = tensorType;
    probabilities.resize(256);
    if (dfl) {
        exponentials.resize(256);
    }
    const int minimum = type == fwk::iface::TensorType::INT8 ? -128 : 0;
    for (size_t index = 0; index < 256; ++index) {
        probabilities[index] =
            Sigmoid((minimum + static_cast<int>(index) - quant.offset) * quant.scale);
        if (dfl) {
            // q_max - q is in [0, 255]; the zero point cancels in softmax.
            exponentials[index] = std::exp(-quant.scale * index);
        }
    }
}

YoloV8NewPostProcess::YoloV8NewPostProcess(
    const std::vector<std::shared_ptr<fwk::iface::TensorIface>>& outputTensors,
    std::vector<yolov8_detection::DetectionResult>& results,
    const YoloV8PostProcessParams& params,
    bool useTensorLayout) :
    m_outputTensors{outputTensors}, m_results{results}, m_params{params},
    m_useTensorLayout{useTensorLayout}
{}

bool YoloV8NewPostProcess::DoPostProcess()
{
    if (!DecodeDetections()) {
        return false;
    }
    SuppressDetections();
    return true;
}

bool YoloV8NewPostProcess::DecodeDetections()
{
    struct FeatureMap {
        OutputReader reader;
        const LookupTables* lookup{nullptr};
        size_t height{0};
        size_t width{0};
        size_t channels{0};
        size_t stride{0};
        size_t angleOffset{0};
        bool channelsFirst{false};
        float minimumClassCode{-std::numeric_limits<float>::infinity()};
    };

    m_results.clear();
    m_candidates.clear();
    m_rotatedNms = false;
    if (m_params.numClasses == 0 || m_params.imageWidth == 0 || m_params.imageHeight == 0) {
        printf_err("Raw YOLOv8 heads require a class count and nonzero image dimensions\n");
        return false;
    }
    size_t dflBins   = 0;
    size_t dflValues = 0;

    if (m_outputTensors.size() != 3 && m_outputTensors.size() != 4) {
        printf_err("New YOLOv8 model must have three or four output tensors\n");
        return false;
    }

    const auto validFeature = [this](size_t height, size_t width, size_t channels) {
        return height > 0 && width > 0 && channels > m_params.numClasses &&
               (channels - m_params.numClasses) % 4 == 0 && m_params.imageWidth % width == 0 &&
               m_params.imageHeight % height == 0 &&
               m_params.imageWidth / width == m_params.imageHeight / height;
    };
    std::shared_ptr<fwk::iface::TensorIface> angleTensor;
    std::vector<FeatureMap> featureMaps;
    featureMaps.reserve(3);

    for (const auto& tensor : m_outputTensors) {
        if (tensor == nullptr || (tensor->Type() != fwk::iface::TensorType::FP32 &&
                                  tensor->Type() != fwk::iface::TensorType::INT8 &&
                                  tensor->Type() != fwk::iface::TensorType::UINT8)) {
            printf_err("New YOLOv8 outputs must be float32, int8 or uint8 tensors\n");
            return false;
        }

        const auto shape = tensor->Shape();
        const bool channelsFirstShape =
            shape.size() == 4 && shape[0] == 1 && validFeature(shape[2], shape[3], shape[1]);
        const bool channelsLastShape =
            shape.size() == 4 && shape[0] == 1 && validFeature(shape[1], shape[2], shape[3]);
        const bool isFeatureShape = channelsFirstShape || channelsLastShape;
        const bool isAngleShape =
            (shape.size() == 1) || (shape.size() == 2 && shape[0] == 1) ||
            (shape.size() == 3 && shape[0] == 1 && (shape[1] == 1 || shape[2] == 1)) ||
            (!isFeatureShape && m_params.orientedBoxes && shape.size() == 4 && shape[0] == 1 &&
             shape[3] == 1 && (shape[1] == 1 || shape[2] == 1));
        if (isAngleShape) {
            if (angleTensor != nullptr) {
                printf_err("New YOLOv8 model has multiple angle outputs\n");
                return false;
            }
            angleTensor = tensor;
            continue;
        }

        if (shape.size() != 4 || shape[0] != 1 || shape[1] == 0 || shape[2] == 0 || shape[3] == 0) {
            printf_err("Invalid new YOLOv8 feature map shape\n");
            return false;
        }

        OutputReader reader;
        if (!reader.Init(tensor)) {
            printf_err("Invalid new YOLOv8 output storage or quantization\n");
            return false;
        }
        if (channelsFirstShape) {
            const bool channelsFirst =
                !m_useTensorLayout || tensor->Layout() != fwk::iface::TensorLayout::NHWC;
            featureMaps.push_back(
                {reader, nullptr, shape[2], shape[3], shape[1], 0, 0, channelsFirst});
        } else if (channelsLastShape) {
            featureMaps.push_back({reader, nullptr, shape[1], shape[2], shape[3], 0, 0, false});
        } else {
            printf_err("New YOLOv8 feature map has no DFL/class channel dimension\n");
            return false;
        }
    }

    if (featureMaps.size() != 3) {
        printf_err("New YOLOv8 model must have three feature maps\n");
        return false;
    }

    std::sort(featureMaps.begin(),
              featureMaps.end(),
              [](const FeatureMap& first, const FeatureMap& second) {
                  return first.width > second.width;
              });

    OutputReader angleReader;
    auto& angleLookup = m_lookups.back();
    if (angleTensor != nullptr) {
        if (!angleReader.Init(angleTensor)) {
            printf_err("Invalid new YOLOv8 angle storage or quantization\n");
            return false;
        }
        if (angleReader.IsQuantized() && m_params.angleIsLogits) {
            angleLookup.Update(angleReader.quant, angleReader.type, false);
        }
    }
    const size_t angleElements = angleTensor == nullptr ? 0 : angleReader.elements;
    size_t angleOffset         = 0;
    const size_t numClasses    = m_params.numClasses;

    size_t featureIndex = 0;
    for (auto& featureMap : featureMaps) {
        if (m_params.imageWidth % featureMap.width != 0 ||
            m_params.imageHeight % featureMap.height != 0 || featureMap.channels <= numClasses) {
            printf_err("Invalid new YOLOv8 feature map dimensions\n");
            return false;
        }

        featureMap.stride = m_params.imageWidth / featureMap.width;
        if (featureMap.stride == 0 ||
            m_params.imageHeight / featureMap.height != featureMap.stride) {
            printf_err("New YOLOv8 feature map stride is invalid\n");
            return false;
        }

        if (featureMap.reader.elements !=
            featureMap.height * featureMap.width * featureMap.channels) {
            printf_err("New YOLOv8 feature map storage size is invalid\n");
            return false;
        }
        const size_t bins = (featureMap.channels - numClasses) / 4;
        if (bins < 2 || bins > 32 || (m_params.dflBins != 0 && m_params.dflBins != bins) ||
            (dflBins != 0 && dflBins != bins)) {
            printf_err("New YOLOv8 feature maps have invalid or inconsistent DFL bins\n");
            return false;
        }
        dflBins   = bins;
        dflValues = 4 * bins;

        if (featureMap.reader.IsQuantized()) {
            auto& lookup = m_lookups[featureIndex];
            lookup.Update(featureMap.reader.quant, featureMap.reader.type, true);
            featureMap.lookup = &lookup;
            const auto first  = std::lower_bound(
                lookup.probabilities.begin(), lookup.probabilities.end(), m_params.scoreThreshold);
            const int minimum = featureMap.reader.type == fwk::iface::TensorType::INT8 ? -128 : 0;
            featureMap.minimumClassCode =
                static_cast<float>(minimum + (first - lookup.probabilities.begin()));
        }
        ++featureIndex;
        featureMap.angleOffset = angleOffset;
        angleOffset += featureMap.height * featureMap.width;
    }

    if (angleTensor != nullptr && angleOffset != angleElements) {
        printf_err("New YOLOv8 angle output has an unexpected size\n");
        return false;
    }

    m_candidates.reserve(angleOffset);
    std::vector<float> dflLogits(dflBins);

    for (const auto& featureMap : featureMaps) {
        for (size_t y = 0; y < featureMap.height; ++y) {
            for (size_t x = 0; x < featureMap.width; ++x) {
                const size_t cell = y * featureMap.width + x;
                // Positive scale and Sigmoid preserve ordering. Keep upstream probability ties.
                float bestLogit = -std::numeric_limits<float>::infinity();
                int bestClass   = -1;
                float bestScore = 0.0F;
                const size_t classStride =
                    featureMap.channelsFirst ? featureMap.height * featureMap.width : 1;
                const size_t firstClass = featureMap.channelsFirst
                                              ? dflValues * classStride + cell
                                              : cell * featureMap.channels + dflValues;
                for (size_t classIndex = 0; classIndex < numClasses; ++classIndex) {
                    float value = 0.0F;
                    if (!featureMap.reader.ReadRaw(firstClass + classIndex * classStride, value)) {
                        printf_err("New YOLOv8 class output is unsupported\n");
                        return false;
                    }
                    if (value > bestLogit) {
                        bestLogit = value;
                        if (value >= featureMap.minimumClassCode) {
                            const float score =
                                featureMap.reader.IsQuantized()
                                    ? featureMap.lookup
                                          ->probabilities[featureMap.reader.LookupIndex(value)]
                                    : Sigmoid(value);
                            if (score > bestScore) {
                                bestScore = score;
                                bestClass = static_cast<int>(classIndex);
                            }
                        }
                    }
                }
                if (angleTensor != nullptr && !m_params.orientedBoxes) {
                    float objectness = 0.0F;
                    if (!angleReader.ReadRaw(featureMap.angleOffset + cell, objectness)) {
                        return false;
                    }
                    bestScore *= Sigmoid(angleReader.ToFloat(objectness));
                }
                if (bestClass < 0 || !(bestScore > 0.0F) || bestScore < m_params.scoreThreshold) {
                    continue;
                }

                float angleRadians = 0.0F;
                if (angleTensor != nullptr && m_params.orientedBoxes) {
                    float angleValue = 0.0F;
                    if (!angleReader.ReadRaw(featureMap.angleOffset + cell, angleValue)) {
                        printf_err("New YOLOv8 angle output is unsupported\n");
                        return false;
                    }
                    constexpr float kPi = 3.14159265358979323846F;
                    angleRadians =
                        m_params.angleIsLogits
                            ? ((angleReader.IsQuantized()
                                    ? angleLookup.probabilities[angleReader.LookupIndex(angleValue)]
                                    : Sigmoid(angleValue)) -
                               0.25F) *
                                  kPi
                            : angleReader.ToFloat(angleValue);
                }

                std::array<float, 4> distances{};

                for (size_t side = 0; side < 4; ++side) {
                    float maximum = -std::numeric_limits<float>::infinity();
                    for (size_t bin = 0; bin < dflBins; ++bin) {
                        const size_t channel = side * dflBins + bin;
                        const size_t index =
                            featureMap.channelsFirst
                                ? channel * featureMap.height * featureMap.width + cell
                                : cell * featureMap.channels + channel;
                        if (!featureMap.reader.ReadRaw(index, dflLogits[bin])) {
                            printf_err("New YOLOv8 feature map output is unsupported\n");
                            return false;
                        }
                        maximum = std::max(maximum, dflLogits[bin]);
                    }

                    float sum         = 0.0F;
                    float weightedSum = 0.0F;
                    for (size_t bin = 0; bin < dflBins; ++bin) {
                        const float probability =
                            featureMap.reader.IsQuantized()
                                ? featureMap.lookup
                                      ->exponentials[static_cast<size_t>(maximum - dflLogits[bin])]
                                : std::exp(dflLogits[bin] - maximum);
                        sum += probability;
                        weightedSum += static_cast<float>(bin) * probability;
                    }
                    distances[side] = sum > 0.0F ? weightedSum / sum : 0.0F;
                }

                const float cosine  = std::cos(angleRadians);
                const float sine    = std::sin(angleRadians);
                const float offsetX = (distances[2] - distances[0]) * 0.5F;
                const float offsetY = (distances[3] - distances[1]) * 0.5F;
                const float centerX =
                    (static_cast<float>(x) + 0.5F + offsetX * cosine - offsetY * sine) *
                    featureMap.stride;
                const float centerY =
                    (static_cast<float>(y) + 0.5F + offsetX * sine + offsetY * cosine) *
                    featureMap.stride;
                const float boxWidth  = (distances[0] + distances[2]) * featureMap.stride;
                const float boxHeight = (distances[1] + distances[3]) * featureMap.stride;
                yolov8_detection::DetectionResult result{
                    bestScore, bestClass, centerX, centerY, boxWidth, boxHeight, angleRadians};
                if (IntersectsImage(result, m_params)) {
                    m_candidates.push_back(result);
                }
            }
        }
    }

    m_rotatedNms = m_params.orientedBoxes && angleTensor != nullptr;
    return true;
}

void YoloV8NewPostProcess::SuppressDetections()
{
    m_results.clear();
    ApplyClassAwareNms(m_candidates, m_results, m_params, m_rotatedNms);
}

} /* namespace arm::app */
