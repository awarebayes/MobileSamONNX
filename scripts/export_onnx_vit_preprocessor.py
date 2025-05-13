# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

import sys

sys.path.append(".")
from mobile_sam.modeling.tiny_vit_sam import TinyViT

import argparse
import warnings
from onnxsim import simplify
import onnx

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

parser = argparse.ArgumentParser(
    description="Export the SAM prompt encoder and mask decoder to an ONNX model."
)
parser.add_argument(
    "--output", type=str, required=True, help="The filename to save the ONNX model to."
)

parser.add_argument(
    "--opset",
    type=int,
    default=17,
    help="The ONNX opset version to use. Must be >=11",
)

parser.add_argument(
    "--width",
    type=int,
    default=1920,
    help="Input image width.",
)

parser.add_argument(
    "--height",
    type=int,
    default=1080,
    help="Input image height.",
)


parser.add_argument(
    "--quantize-out",
    type=str,
    default=None,
    help=(
        "If set, will quantize the model and save it with this name. "
        "Quantization is performed with quantize_dynamic from onnxruntime.quantization.quantize."
    ),
)


class PreprocessModule(nn.Module):
    def __init__(self,
            pixel_mean: list[float] = [123.675, 116.28, 103.53],
            pixel_std: list[float] = [58.395, 57.12, 57.375],
            target_size=1024
        ):
        super().__init__()
        
        # Register as buffers to include them in ONNX
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(1, 3, 1, 1))
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input (H, W, C) in [0, 255] BGR (uint8)

        # CHW
        x = x.permute(2, 0, 1).contiguous()[None, :, :, :]

        # Convert to float32
        x = x.float() 

        # BGR to RGB (swap channels 0 and 2)
        x = x[:, [2, 1, 0], :, :]

        # Normalize
        x = (x - self.pixel_mean) / self.pixel_std

        # Resize
        h, w = x.shape[-2], x.shape[-1]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize using torch's interpolate (ONNX-compatible)
        x = torch.nn.functional.interpolate(
            x,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )

        # Pad to square (assuming x is CHW format)
        padh = self.target_size - new_h 
        padw = self.target_size - new_w 
        x = torch.nn.functional.pad(x, (0, padw, 0, padh))  # (left, right, top, bottom)
        return x
    
def run_export(
    output: str,
    opset: int,
    width: int,
    height: int
):
    onnx_model = PreprocessModule()
    image = torch.randint(0, 255, (height, width, 3), dtype=torch.uint8)
    _ = onnx_model.forward(image)

    output_names = ["preprocessed_image"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(output, "wb") as f:
            print(f"Exporting onnx model to {output}...")
            torch.onnx.export(
                onnx_model,
                tuple([image,]),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=["image"],
                output_names=output_names,
                dynamic_axes=None,
            )
        model = onnx.load(output)
        model_simp, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
        with open(output, "wb") as f:
            onnx.save_model(model_simp, output)

    if onnxruntime_exists:
        ort_inputs = {"image": to_numpy(image)}
        providers = ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(output, providers=providers)
        _ = ort_session.run(None, ort_inputs)
        print("Model has successfully been run with ONNXRuntime.")


def to_numpy(tensor):
    return tensor.cpu().numpy()

if __name__ == "__main__":
    args = parser.parse_args()
    run_export(
        output=args.output,
        opset=args.opset,
        width=args.width,
        height=args.height,
    )


