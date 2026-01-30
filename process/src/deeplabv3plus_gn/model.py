"""
The MIT License

Copyright (c) 2019, Pavel Yakubovskiy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

----------------------------------------------------------------------------------------

Cloned from

Repo: https://github.com/qubvel/segmentation_models.pytorch
Commit: 35d79c1aa5fb26ba0b2c1ec67084c66d43687220

Modified from the above by removing DeepLabV3, and replacing DeepLabV3PlusDecoder with
DeeplabV3plusGNDecoder decoder (also changed name from DeepLabV3Plus to DeepLabV3PlusGN)
"""

from typing import Optional
from .decoder import DeepLabV3PlusGNDecoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from encoders_init import get_encoder


class DeepLabV3PlusGN(SegmentationModel):
    """DeepLabV3+ implementation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder
            (a.k.a backbone) to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage
            generate features two times smaller in spatial dimensions than previous one
            (e.g. for depth 0 we will have features with shapes [(N, C, H, W),], for
            depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on). Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"**
            (pre-training on ImageNet) and other pretrained weights (see table with
            available weights for each encoder_name)
        encoder_output_stride: Downsampling factor for last encoder features (see
            original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be a tuple of 3
            integer values)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of
            channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**,
            **"tanh"**, **"identity"**, **callable** and **None**.  Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve input-output
        spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification
            head). Auxiliary output is build on top of encoder if **aux_params** is not
            **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                  (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3PlusGN**

    Reference:
        https://arxiv.org/abs/1802.02611v3

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: int = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36),
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError(
                "Encoder output stride should be 8 or 16, got {}".format(
                    encoder_output_stride
                )
            )

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )

        self.decoder = DeepLabV3PlusGNDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None
