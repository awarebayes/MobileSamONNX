# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

import sys

sys.path.append(".")
from mobile_sam.modeling.tiny_vit_sam import TinyViT
from mobile_sam import sam_model_registry
from mobile_sam.utils.onnx import SamOnnxModel

import argparse
import warnings

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

parser = argparse.ArgumentParser(
    description="Export the SAM prompt encoder and mask decoder to an ONNX model."
)

parser.add_argument(
    "--checkpoint", type=str, required=True, help="The path to the SAM model checkpoint."
)

parser.add_argument(
    "--output", type=str, required=True, help="The filename to save the ONNX model to."
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="In ['default', 'vit_h', 'vit_l', 'vit_b']. Which type of SAM model to export.",
)

parser.add_argument(
    "--opset",
    type=int,
    default=16,
    help="The ONNX opset version to use. Must be >=11",
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

def run_export(
    model_type: str,
    checkpoint: str,
    output: str,
    opset: int,
):
    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    onnx_model = sam.image_encoder
    image = torch.randn((1, 3, 1024, 1024))
    _ = onnx_model.forward(image)

    output_names = ["embeddings"]

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
                input_names=["preprocessed_image"],
                output_names=output_names,
                dynamic_axes=None,
            )

    if onnxruntime_exists:
        ort_inputs = {"preprocessed_image": to_numpy(image)}
        providers = ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(output, providers=providers)
        _ = ort_session.run(None, ort_inputs)
        print("Model has successfully been run with ONNXRuntime.")


def to_numpy(tensor):
    return tensor.cpu().numpy()

if __name__ == "__main__":
    args = parser.parse_args()
    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output=args.output,
        opset=args.opset,
    )


