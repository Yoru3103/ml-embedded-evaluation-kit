/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or
 * its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mlek/fwk/executorch/EtTensor.hpp"

namespace arm::app::fwk::et {

EtTensor::EtTensor(executorch::aten::TensorImpl* tensorImpl, iface::QuantParams quantParams) :
    m_tensor(tensorImpl), m_quantParams(quantParams)
{}

EtTensor::EtTensor(executorch::aten::Tensor& tensor, iface::QuantParams quantParams) :
    m_tensor(tensor), m_quantParams(quantParams)
{}

void* EtTensor::GetData()
{ return this->m_tensor.mutable_data_ptr<void>(); }

size_t EtTensor::Bytes()
{ return this->m_tensor.nbytes(); }

size_t EtTensor::GetNumElements()
{ return this->m_tensor.numel(); }

std::vector<size_t> EtTensor::Shape()
{
    auto sz = this->m_tensor.sizes();
    return std::vector<size_t>(sz.begin(), sz.end());
}

iface::TensorType EtTensor::Type()
{
    switch (this->m_tensor.dtype()) {
    case executorch::aten::ScalarType::Char:
    case executorch::aten::ScalarType::QInt8:
        return iface::TensorType::INT8;
    case executorch::aten::ScalarType::Byte:
    case executorch::aten::ScalarType::QUInt8:
        return iface::TensorType::UINT8;
    case executorch::aten::ScalarType::Short:
        return iface::TensorType::INT16;
    case executorch::aten::ScalarType::Int:
        return iface::TensorType::INT32;
    case executorch::aten::ScalarType::Float:
        return iface::TensorType::FP32;
    default:
        return iface::TensorType::INVALID;
    }
}

iface::TensorLayout EtTensor::Layout()
{
    const auto dimOrder = this->m_tensor.dim_order();
    if (dimOrder.size() != 4) {
        return iface::TensorLayout::INVALID;
    }

    if (dimOrder[0] == 0 && dimOrder[1] == 1 && dimOrder[2] == 2 && dimOrder[3] == 3) {
        return iface::TensorLayout::NCHW;
    }
    if (dimOrder[0] == 0 && dimOrder[1] == 2 && dimOrder[2] == 3 && dimOrder[3] == 1) {
        return iface::TensorLayout::NHWC;
    }
    return iface::TensorLayout::INVALID;
}

iface::QuantParams EtTensor::GetQuantParams()
{ return this->m_quantParams; }

} // namespace arm::app::fwk::et
