/* SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 */
  // Specialize only the validated in-place KV state contract. All other
  // clamp calls continue through the original portable implementation.
  if (in.scalar_type() == ScalarType::Short &&
      out.scalar_type() == ScalarType::Short &&
      min_opt.has_value() && max_opt.has_value() &&
      min_opt->isIntegral(false) && max_opt->isIntegral(false) &&
      min_opt->to<int64_t>() == -32767 && max_opt->to<int64_t>() == 32767 &&
      in.dim() == 4 && out.dim() == 4 &&
      in.size(0) == 1 && in.size(1) == 8 && in.size(2) == 64 && in.size(3) == 64 &&
      out.size(0) == 1 && out.size(1) == 8 && out.size(2) == 64 && out.size(3) == 64 &&
      tensors_have_same_dim_order(in, out) &&
      in.const_data_ptr<int16_t>() == out.mutable_data_ptr<int16_t>()) {
    arm::app::tinystories::ClampCache(out.mutable_data_ptr<int16_t>(), out.numel());
    return out;
  }
