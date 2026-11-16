/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/asr/AsrClassifier.hpp"
#include <stdexcept>

using namespace arm::app;

class TestOutputTensor : public fwk::iface::TensorIface {
public:
    std::vector<int8_t> data = std::vector<int8_t>(148 * 29, -128);
    void* GetData() override
    { return data.data(); }
    size_t Bytes() override
    { return data.size(); }
    size_t GetNumElements() override
    { return data.size(); }
    fwk::iface::TensorType Type() override
    { return fwk::iface::TensorType::INT8; }
    fwk::iface::TensorLayout Layout() override
    { return fwk::iface::TensorLayout::NHWC; }
    std::vector<size_t> Shape() override
    { return {1, 1, 148, 29}; }
    fwk::iface::QuantParams GetQuantParams() override
    { return {1.0f / 256, -128}; }
};

int main()
{
    auto tensor = std::make_shared<TestOutputTensor>();
    std::vector<std::string> labels;
    for (int i = 0; i < 29; ++i) {
        labels.push_back(std::to_string(i));
    }
    for (size_t row = 2; row < 148; ++row) {
        tensor->data[row * 29 + (row * 7) % 29] = static_cast<int8_t>(static_cast<int>(row) - 128);
    }
    tensor->data[3] = tensor->data[8] = 127; // A tie must retain the first winning class.
    AsrClassifier classifier(1, 2, 2, 3);
    std::vector<ClassificationResult> results;
    if (!classifier.GetClassificationResults(tensor, results, labels, 1) || results.size() != 148) {
        throw std::runtime_error("Classifier must return all output rows");
    }
    for (size_t row = 0; row < results.size(); ++row) {
        const size_t index       = row == 0 ? 3 : row == 1 ? 0 : (row * 7) % 29;
        const double probability = row == 0 ? 255.0 / 256 : row == 1 ? 0 : row / 256.0;
        if (results[row].m_labelIdx != index || results[row].m_label != labels[index] ||
            results[row].m_normalisedVal != probability) {
            throw std::runtime_error(
                "Timing mode must preserve INT8 argmax, tie-breaking and scores");
        }
    }
}
