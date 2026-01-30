#
# The MIT License
#
# Copyright (c) 2019, Pavel Yakubovskiy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

"""
Cloned from segmentation_models_pytorch/encoders/__init__.py
commit 35d79c1aa5fb26ba0b2c1ec67084c66d43687220

Change:
    - Change import location from local (wrt. segmentation_models_pytorch) to global
      (wrt. this repo). Example:
      < from .vgg import vgg_encoders
      > from segmentation_models_pytorch.encoders.vgg import vgg_encoders
    - Add local timm_nfnet_encoders
    - Add MIT license from original repo on top of file
    - Black reformat
    - Option for custom preprocessing params
    - Type hints
"""

import functools
from typing import Any, Callable, Dict, Optional, Sequence
import torch.utils.model_zoo as model_zoo

from segmentation_models_pytorch.encoders.resnet import resnet_encoders
from segmentation_models_pytorch.encoders.dpn import dpn_encoders
from segmentation_models_pytorch.encoders.vgg import vgg_encoders
from segmentation_models_pytorch.encoders.senet import senet_encoders
from segmentation_models_pytorch.encoders.densenet import densenet_encoders
from segmentation_models_pytorch.encoders.inceptionresnetv2 import (
    inceptionresnetv2_encoders,
)
from segmentation_models_pytorch.encoders.inceptionv4 import inceptionv4_encoders
from segmentation_models_pytorch.encoders.efficientnet import efficient_net_encoders
from segmentation_models_pytorch.encoders.mobilenet import mobilenet_encoders
from segmentation_models_pytorch.encoders.xception import xception_encoders
from segmentation_models_pytorch.encoders.timm_efficientnet import (
    timm_efficientnet_encoders,
)
from segmentation_models_pytorch.encoders.timm_resnest import timm_resnest_encoders
from segmentation_models_pytorch.encoders.timm_res2net import timm_res2net_encoders
from segmentation_models_pytorch.encoders.timm_regnet import timm_regnet_encoders
from segmentation_models_pytorch.encoders.timm_sknet import timm_sknet_encoders
from segmentation_models_pytorch.encoders.timm_mobilenetv3 import (
    timm_mobilenetv3_encoders,
)
from segmentation_models_pytorch.encoders.timm_gernet import timm_gernet_encoders

from segmentation_models_pytorch.encoders.timm_universal import TimmUniversalEncoder

from segmentation_models_pytorch.encoders._preprocessing import preprocess_input

from timm_nfnet_encoder import timm_nfnet_encoders

encoders = {}
encoders.update(resnet_encoders)
encoders.update(dpn_encoders)
encoders.update(vgg_encoders)
encoders.update(senet_encoders)
encoders.update(densenet_encoders)
encoders.update(inceptionresnetv2_encoders)
encoders.update(inceptionv4_encoders)
encoders.update(efficient_net_encoders)
encoders.update(mobilenet_encoders)
encoders.update(xception_encoders)
encoders.update(timm_efficientnet_encoders)
encoders.update(timm_resnest_encoders)
encoders.update(timm_res2net_encoders)
encoders.update(timm_regnet_encoders)
encoders.update(timm_sknet_encoders)
encoders.update(timm_mobilenetv3_encoders)
encoders.update(timm_gernet_encoders)
encoders.update(timm_nfnet_encoders)


def get_encoder(
    name: str,
    in_channels: int = 3,
    depth: int = 5,
    weights: Optional[str] = None,
    output_stride: int = 32,
    **kwargs: Any,
):

    if name.startswith("tu-"):
        name = name[3:]
        encoder = TimmUniversalEncoder(
            name=name,
            in_channels=in_channels,
            depth=depth,
            output_stride=output_stride,
            pretrained=weights is not None,
            **kwargs,
        )
        return encoder

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError(
            "Wrong encoder name `{}`, supported encoders: {}".format(
                name, list(encoders.keys())
            )
        )

    params = encoders[name]["params"]
    params.update(depth=depth)
    params.update(output_stride=output_stride)
    encoder = Encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError(
                f"Wrong pretrained weights `{weights}` for encoder `{name}`."
                "Available options are: {}".format(
                    list(encoders[name]["pretrained_settings"].keys()),
                )
            )
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels, pretrained=weights is not None)
    # if output_stride != 32:
    #     encoder.make_dilated(output_stride)

    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_params(
    encoder_name: str,
    pretrained: Optional[str] = "imagenet",
    input_space: Optional[str] = None,
    input_range: Optional[Sequence[float]] = None,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    if pretrained is not None:
        encoder_name = encoder_name.replace("tu-", "timm-").replace("_", "-")
        settings = encoders[encoder_name]["pretrained_settings"]
        if pretrained not in settings.keys():
            raise ValueError("Available pretrained options {}".format(settings.keys()))
        input_space = settings[pretrained].get("input_space")
        input_range = settings[pretrained].get("input_range")
        mean = settings[pretrained].get("mean")
        std = settings[pretrained].get("std")

    formatted_settings = {
        "input_space": input_space,
        "input_range": input_range,
        "mean": mean,
        "std": std,
    }
    return formatted_settings


def get_preprocessing_fn(
    encoder_name: str,
    pretrained: Optional[str] = "imagenet",
    input_space: Optional[str] = None,
    input_range: Optional[Sequence[float]] = None,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> Callable:
    params = get_preprocessing_params(
        encoder_name, pretrained, input_space, input_range, mean, std
    )
    return functools.partial(preprocess_input, **params)
