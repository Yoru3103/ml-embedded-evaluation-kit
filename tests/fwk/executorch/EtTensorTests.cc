/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mlek/fwk/executorch/EtTensor.hpp"

#include <catch.hpp>
#include <cstdint>

TEST_CASE("ExecuTorch integer tensors retain per-tensor quantization", "[executorch]")
{
    using arm::app::fwk::et::EtTensor;
    using arm::app::fwk::iface::TensorType;
    using executorch::aten::ScalarType;
    using executorch::aten::TensorImpl;

    TensorImpl::SizesType sizes[]    = {1, 4};
    TensorImpl::DimOrderType order[] = {0, 1};
    int8_t data[]                    = {-128, -1, 0, 127};
    for (const auto scalarType : {ScalarType::Char, ScalarType::Byte}) {
        TensorImpl impl(scalarType, 2, sizes, data, order);
        const int offset = scalarType == ScalarType::Char ? -128 : 128;
        EtTensor tensor(&impl, {0.0039F, offset});
        CHECK(tensor.Type() ==
              (scalarType == ScalarType::Char ? TensorType::INT8 : TensorType::UINT8));
        CHECK(tensor.GetData() == data);
        CHECK(tensor.Bytes() == 4);
        CHECK(tensor.GetQuantParams().scale == Approx(0.0039F));
        CHECK(tensor.GetQuantParams().offset == offset);
    }
    TensorImpl impl(ScalarType::Char, 2, sizes, data, order);
    executorch::aten::Tensor value(&impl);
    EtTensor output(value, {0.25F, 78});
    CHECK(output.Type() == TensorType::INT8);
    CHECK(output.GetQuantParams().scale == Approx(0.25F));
    CHECK(output.GetQuantParams().offset == 78);
}

TEST_CASE("ExecuTorch floating tensor defaults remain unquantized", "[executorch]")
{
    executorch::aten::TensorImpl::SizesType sizes[] = {1};
    float data[]                                    = {0.5F};
    executorch::aten::TensorImpl impl(executorch::aten::ScalarType::Float, 1, sizes, data);
    arm::app::fwk::et::EtTensor tensor(&impl);
    CHECK(tensor.Type() == arm::app::fwk::iface::TensorType::FP32);
    CHECK(tensor.Bytes() == sizeof(float));
    CHECK(tensor.GetQuantParams().scale == 0);
}
