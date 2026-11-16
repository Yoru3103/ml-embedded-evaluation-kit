/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/use_case/asr/ConformerProcessing.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using arm::app::fwk::iface::QuantParams;
using arm::app::fwk::iface::TensorIface;
using arm::app::fwk::iface::TensorLayout;
using arm::app::fwk::iface::TensorType;

class GuardedTensor : public TensorIface {
public:
    GuardedTensor(std::vector<size_t> shape, TensorType type) :
        m_shape(std::move(shape)), m_type(type), m_data(GetNumElements() * sizeof(float) + 8, 0x7B)
    { std::memset(GetData(), 0, Bytes()); }
    void* GetData() override
    { return m_data.data() + 4; }
    size_t Bytes() override
    { return GetNumElements() * sizeof(float); }
    size_t GetNumElements() override
    {
        return std::accumulate(
            m_shape.begin(), m_shape.end(), size_t{1}, std::multiplies<size_t>());
    }
    TensorType Type() override
    { return m_type; }
    TensorLayout Layout() override
    { return TensorLayout::NHWC; }
    std::vector<size_t> Shape() override
    { return m_shape; }
    QuantParams GetQuantParams() override
    { return {}; }
    bool GuardsIntact() const
    {
        for (size_t i = 0; i < 4; ++i) {
            if (m_data[i] != 0x7B || m_data[m_data.size() - 1 - i] != 0x7B) {
                return false;
            }
        }
        return true;
    }

private:
    std::vector<size_t> m_shape;
    TensorType m_type;
    std::vector<uint8_t> m_data;
};

void Require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}
} // namespace

int main()
{
    auto mel = std::make_shared<GuardedTensor>(std::vector<size_t>{1, 700, 80}, TensorType::FP32);
    auto length = std::make_shared<GuardedTensor>(std::vector<size_t>{1}, TensorType::INT32);
    arm::app::ConformerPreProcess<int16_t> process(mel, length, 512, 160, 700);
    std::vector<int16_t> audio(512, 0);
    Require(process.DoPreProcess(audio.data(), audio.size()), "One frame must succeed");
    int32_t actualLength = 0;
    std::memcpy(&actualLength, length->GetData(), sizeof(actualLength));
    Require(actualLength == 700, "Length must be INT32 700, not float bits");
    auto values = static_cast<float*>(mel->GetData());
    for (size_t i = 0; i < 80; ++i) {
        Require(std::abs(values[i] - std::log(1e-6f)) < 1e-4f, "Silent frame log energy");
    }
    for (size_t i = 80; i < mel->GetNumElements(); ++i) {
        Require(values[i] == -20.f, "Unused frames must be padded with -20");
    }
    Require(mel->GuardsIntact() && length->GuardsIntact(), "Tensor guards changed");
    Require(!process.DoPreProcess(nullptr, 512), "Null input must fail");
    Require(!process.DoPreProcess(audio.data(), 511), "Incomplete frame must fail");

    auto shortMel =
        std::make_shared<GuardedTensor>(std::vector<size_t>{1, 3, 80}, TensorType::FP32);
    arm::app::ConformerPreProcess<int16_t> shortProcess(shortMel, length, 512, 160, 3);
    audio.resize(512 + 2 * 160);
    Require(shortProcess.DoPreProcess(audio.data(), audio.size()), "Exact capacity must fit");
    audio.resize(512 + 3 * 160);
    Require(!shortProcess.DoPreProcess(audio.data(), audio.size()), "Extra frame must fail");
    Require(shortMel->GuardsIntact(), "Oversized input corrupted output");

    std::vector<float> smallOutput(81, 123.f);
    Require(process.GenerateMelSpectrogram(audio.data(), 512, smallOutput.data(), 79) == 0,
            "Partial output frame must fail");
    Require(smallOutput.back() == 123.f, "Undersized output corrupted sentinel");
    Require(process.GenerateMelSpectrogram(audio.data(), 512, smallOutput.data(), 80) == 80,
            "Output capacity must be measured in elements");
    Require(smallOutput.back() == 123.f, "Exact output changed sentinel");

    auto floatLength = std::make_shared<GuardedTensor>(std::vector<size_t>{1}, TensorType::FP32);
    arm::app::ConformerPreProcess<int16_t> wrongType(mel, floatLength, 512, 160, 700);
    Require(!wrongType.DoPreProcess(audio.data(), 512), "FP32 length tensor must fail");
    arm::app::ConformerPreProcess<int16_t> wrongShape(mel, length, 512, 160, 1500);
    Require(!wrongShape.DoPreProcess(audio.data(), 512), "Mismatched frame count must fail");
    arm::app::ConformerPreProcess<int16_t> zeroHop(mel, length, 512, 0, 700);
    Require(!zeroHop.DoPreProcess(audio.data(), 512), "Zero hop must fail");
    Require(process.DoPreProcess(audio.data(), 512), "Repeated inference preprocessing must work");
    std::cout << "Conformer input type, padding, capacity and repeat checks passed\n";
    return 0;
}
