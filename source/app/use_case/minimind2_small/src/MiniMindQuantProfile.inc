/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "MiniMindProfiler.hpp"
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <optional>

namespace arm::app::minimind2_small {
using executorch::runtime::KernelRuntimeContext;
using executorch::aten::Tensor;
using executorch::aten::ScalarType;

// These symbols match the scalar out kernels linked by the Arm LLVM build.
Tensor& RealQuantize(KernelRuntimeContext& context, const Tensor& input,
                  double scale, int64_t zero, int64_t min, int64_t max, ScalarType type,
                  Tensor& output)
    asm("__real__ZN5torch8executor6native23quantize_per_tensor_outERN10executorch7runtime20KernelRuntimeContextERKNS3_7etensor6TensorEdxxxNS6_10ScalarTypeERS7_");
Tensor& WrapQuantize(KernelRuntimeContext& context, const Tensor& input,
                  double scale, int64_t zero, int64_t min, int64_t max, ScalarType type,
                  Tensor& output)
    asm("__wrap__ZN5torch8executor6native23quantize_per_tensor_outERN10executorch7runtime20KernelRuntimeContextERKNS3_7etensor6TensorEdxxxNS6_10ScalarTypeERS7_");
Tensor& WrapQuantize(KernelRuntimeContext& context, const Tensor& input,
                  double scale, int64_t zero, int64_t min, int64_t max, ScalarType type,
                  Tensor& output)
{
    const auto begin = CpuCycles();
    auto& result = RealQuantize(context, input, scale, zero, min, max, type,
                         output);
    GetProfiler().RecordConversion(false, begin, CpuCycles(), input.numel());
    return result;
}

Tensor& RealDequantize(KernelRuntimeContext& context, const Tensor& input,
                  double scale, int64_t zero, int64_t min, int64_t max, ScalarType type,
                  std::optional<ScalarType> outputType, Tensor& output)
    asm("__real__ZN5torch8executor6native25dequantize_per_tensor_outERN10executorch7runtime20KernelRuntimeContextERKNS3_7etensor6TensorEdxxxNS6_10ScalarTypeENSt3__28optionalISA_EERS7_");
Tensor& WrapDequantize(KernelRuntimeContext& context, const Tensor& input,
                  double scale, int64_t zero, int64_t min, int64_t max, ScalarType type,
                  std::optional<ScalarType> outputType, Tensor& output)
    asm("__wrap__ZN5torch8executor6native25dequantize_per_tensor_outERN10executorch7runtime20KernelRuntimeContextERKNS3_7etensor6TensorEdxxxNS6_10ScalarTypeENSt3__28optionalISA_EERS7_");
Tensor& WrapDequantize(KernelRuntimeContext& context, const Tensor& input,
                  double scale, int64_t zero, int64_t min, int64_t max, ScalarType type,
                  std::optional<ScalarType> outputType, Tensor& output)
{
    const auto begin = CpuCycles();
    auto& result = RealDequantize(context, input, scale, zero, min, max, type,
                         outputType, output);
    GetProfiler().RecordConversion(true, begin, CpuCycles(), input.numel());
    return result;
}

Tensor& RealCortexQuantize(KernelRuntimeContext& context, const Tensor& input,
                  double scale, int64_t zero, int64_t min, int64_t max, ScalarType type,
                  Tensor& output)
    asm("__real__ZN8cortex_m6native23quantize_per_tensor_outERN10executorch7runtime20KernelRuntimeContextERKNS2_7etensor6TensorEdxxxNS5_10ScalarTypeERS6_");
Tensor& WrapCortexQuantize(KernelRuntimeContext& context, const Tensor& input,
                  double scale, int64_t zero, int64_t min, int64_t max, ScalarType type,
                  Tensor& output)
    asm("__wrap__ZN8cortex_m6native23quantize_per_tensor_outERN10executorch7runtime20KernelRuntimeContextERKNS2_7etensor6TensorEdxxxNS5_10ScalarTypeERS6_");
Tensor& WrapCortexQuantize(KernelRuntimeContext& context, const Tensor& input,
                  double scale, int64_t zero, int64_t min, int64_t max, ScalarType type,
                  Tensor& output)
{
    const auto begin = CpuCycles();
    auto& result = RealCortexQuantize(context, input, scale, zero, min, max, type,
                         output);
    GetProfiler().RecordConversion(false, begin, CpuCycles(), input.numel());
    return result;
}

Tensor& RealCortexDequantize(KernelRuntimeContext& context, const Tensor& input,
                  double scale, int64_t zero, int64_t min, int64_t max, ScalarType type,
                  Tensor& output)
    asm("__real__ZN8cortex_m6native25dequantize_per_tensor_outERN10executorch7runtime20KernelRuntimeContextERKNS2_7etensor6TensorEdxxxNS5_10ScalarTypeERS6_");
Tensor& WrapCortexDequantize(KernelRuntimeContext& context, const Tensor& input,
                  double scale, int64_t zero, int64_t min, int64_t max, ScalarType type,
                  Tensor& output)
    asm("__wrap__ZN8cortex_m6native25dequantize_per_tensor_outERN10executorch7runtime20KernelRuntimeContextERKNS2_7etensor6TensorEdxxxNS5_10ScalarTypeERS6_");
Tensor& WrapCortexDequantize(KernelRuntimeContext& context, const Tensor& input,
                  double scale, int64_t zero, int64_t min, int64_t max, ScalarType type,
                  Tensor& output)
{
    const auto begin = CpuCycles();
    auto& result = RealCortexDequantize(context, input, scale, zero, min, max, type,
                         output);
    GetProfiler().RecordConversion(true, begin, CpuCycles(), input.numel());
    return result;
}

#ifdef MINIMIND_INT16_KV
using ClampBound = std::optional<executorch::aten::Scalar>;

Tensor& RealClamp(KernelRuntimeContext&, const Tensor&, const ClampBound&, const ClampBound&,
                  Tensor&) asm("__real__ZN5torch8executor6native9clamp_outERN10executorch7runtime20KernelRuntimeContextERKNS3_7etensor6TensorERKNSt3__28optionalINS6_6ScalarEEESF_RS7_");
Tensor& WrapClamp(KernelRuntimeContext&, const Tensor&, const ClampBound&, const ClampBound&,
                  Tensor&) asm("__wrap__ZN5torch8executor6native9clamp_outERN10executorch7runtime20KernelRuntimeContextERKNS3_7etensor6TensorERKNSt3__28optionalINS6_6ScalarEEESF_RS7_");

static bool IsCacheTensor(const Tensor& tensor)
{
    if (tensor.scalar_type() != ScalarType::Short || tensor.dim() != 4 ||
        tensor.size(0) != 1 || tensor.size(1) != 2 || tensor.size(2) != kContextLength ||
        tensor.size(3) != 64) {
        return false;
    }
    const auto strides = tensor.strides();
    return strides.size() == 4 && strides[1] == kContextLength * 64 &&
           strides[2] == 64 && strides[3] == 1;
}

Tensor& WrapClamp(KernelRuntimeContext& context, const Tensor& input, const ClampBound& min,
                  const ClampBound& max, Tensor& output)
{
    const auto begin = CpuCycles();
    const bool eligible = IsCacheTensor(input) && IsCacheTensor(output) &&
                          min && max && min->isIntegral(false) && max->isIntegral(false) &&
                          min->to<int64_t>() == -32767 && max->to<int64_t>() == 32767;
    auto* data = eligible ? output.mutable_data_ptr<int16_t>() : nullptr;
    const bool fast = data && data == input.const_data_ptr<int16_t>();
    if (fast) {
        for (size_t i = 0; i < static_cast<size_t>(output.numel()); ++i) {
            if (data[i] == INT16_MIN) {
                data[i] = -32767;
            }
        }
    } else {
        RealClamp(context, input, min, max, output);
    }
    GetProfiler().RecordClamp(fast, CpuCycles() - begin, output.numel());
    return output;
}
#endif
} // namespace arm::app::minimind2_small
