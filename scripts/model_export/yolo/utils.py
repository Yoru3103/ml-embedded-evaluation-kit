# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Detection-model adaptation and host output checks for YOLO PTE export."""

from pathlib import Path
import types

import torch
from ultralytics import YOLO


# 返回boxes与logits
def split_inference(head: torch.nn.Module, predictions: dict) -> tuple:
    """Decode boxes without combining their quantization range with class logits."""
    return getattr(head, "_get_decode_boxes")(predictions), predictions["scores"]


class ExportYolo(torch.nn.Module):
    """Return one contiguous tensor of normalized XYWH and class probabilities."""

    def __init__(self, model: torch.nn.Module, image_size: int):
        """Store the detection model and its fixed square input size."""
        super().__init__()
        self.model = model
        self.image_size = image_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize boxes before concatenating them with probabilities."""
        boxes, logits = self.model(inputs)      # 修改后的检测头分别输出像素坐标框、类别logits
        # 框坐标归一化，logits转成类别概率，最终拼接成一个连续张量[1, 4+C, N]
        return torch.cat((boxes / self.image_size, logits.sigmoid()), dim=1).contiguous()


def load_model(path: Path, image_size: int, example: torch.Tensor) -> tuple:
    """Adapt a supported Detect head and verify parity with its original output."""
    yolo = YOLO(str(path))                      # 加载 Ultralytics 权重
    model = yolo.model.cpu().float().eval()
    head = model.model[-1]                      # 取模型结构中的最后一个模块（YOLO的检测头）
    # 检查模型类型，如果是分类、分割等任务，就拒绝处理
    # 读取检测头的 end2end 属性；没有这个属性时返回默认值 False。如果属性为 True，也拒绝处理。
    if yolo.task != "detect" or getattr(head, "end2end", False):
        raise ValueError("Only non-end-to-end Ultralytics detection checkpoints are supported")
    # 检查框解码接口是否存在
    if not callable(getattr(head, "_get_decode_boxes", None)):
        raise ValueError("This Ultralytics Detect version lacks _get_decode_boxes")
    head.export = True
    with torch.no_grad():       # 不记录梯度，运行原始模型
        original = model(example)   # 保存检测头尚未被修改时的输出
    setattr(head, "_inference", types.MethodType(split_inference, head))
    wrapper = ExportYolo(model, image_size).eval()
    with torch.no_grad():
        output = wrapper(example)
        boxes, probabilities = output[:, :4], output[:, 4:]
        if boxes.shape[1] != 4 or probabilities.shape[1] != head.nc:
            raise ValueError("Unexpected detection output layout")
        torch.testing.assert_close(boxes * image_size, original[:, :4])
        torch.testing.assert_close(probabilities, original[:, 4:])
    return wrapper, model.names






def compare_outputs(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    """Compare probabilities at identical anchors, including the strongest detection."""
    if reference.shape != actual.shape or not torch.isfinite(actual).all():
        raise ValueError("Output shape mismatch or non-finite inference result")
    boxes, probabilities = reference[:, :4], reference[:, 4:]
    quant_boxes, quant_probabilities = actual[:, :4], actual[:, 4:]
    if quant_probabilities.min() < 0 or quant_probabilities.max() > 1:
        raise ValueError("Quantized probabilities are outside [0, 1]")
    best = int(probabilities.flatten().argmax())
    return {
        "float_max_probability": float(probabilities.max()),
        "quantized_max_probability": float(quant_probabilities.max()),
        "quantized_probability_at_float_best": float(quant_probabilities.flatten()[best]),
        "best_probability_drop": float(probabilities.flatten()[best]
                                       - quant_probabilities.flatten()[best]),
        "probability_mae": float((probabilities - quant_probabilities).abs().mean()),
        "box_mae_normalized": float((boxes - quant_boxes).abs().mean()),
        "float_probability_range": [float(probabilities.min()), float(probabilities.max())],
        "quantized_probability_range": [float(quant_probabilities.min()),
                                        float(quant_probabilities.max())],
    }
