"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------------------------
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

Convolution with Weight Standardization (StdConv and ScaledStdConv)

StdConv:
@article{weightstandardization,
  author    = {Siyuan Qiao and Huiyu Wang and Chenxi Liu and Wei Shen and Alan Yuille},
  title     = {Weight Standardization},
  journal   = {arXiv preprint arXiv:1903.10520},
  year      = {2019},
}
Code:
- https://github.com/joe-siyuan-qiao/WeightStandardization

ScaledStdConv:
Paper:
- Characterizing signal propagation to close the performance gap in unnormalized ResNets
- https://arxiv.org/abs/2101.08692
Official Deepmind JAX code:
- https://github.com/deepmind/deepmind-research/tree/master/nfnets

Hacked together by / copyright Ross Wightman, 2021.

----------------------------------------------------------------------------------------

Cloned from

Repo: https://github.com/qubvel/segmentation_models.pytorch
Commit: 35d79c1aa5fb26ba0b2c1ec67084c66d43687220

Copy of deeplabv3plus_gn without group norm and without ASPPPooling.
Convolutions are replaced with scaled std convolutions
"""

import torch
from torch import nn

__all__ = ["DeepLabV3PlusStdConvDecoder"]


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    """Calculate symmetric padding for a convolution"""
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class ScaledStdConv2d(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization.

    Paper:
    - Characterizing signal propagation to close the performance gap in unnormalized
    ResNets
    - https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl.
    The impact is minor.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=True,
        gamma=1.0,
        eps=1e-6,
        gain_init=1.0,
    ):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5  # gamma * 1 / sqrt(fan-in)
        self.eps = eps

    def forward(self, x):
        weight = torch.nn.functional.batch_norm(
            self.weight.reshape(1, self.out_channels, -1),
            None,
            None,
            weight=(self.gain * self.scale).view(-1),
            training=True,
            momentum=0.0,
            eps=self.eps,
        ).reshape_as(self.weight)
        return torch.nn.functional.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class DeepLabV3PlusStdConvDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        out_channels=256,
        atrous_rates=(12, 24, 36),
        output_stride=16,
    ):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError(
                "Output stride should be 8 or 16, got {}.".format(output_stride)
            )
        group_size = 8  # 1 is 'instance norm'. Number of layer chans is 'layer norm'
        highres_out_channels = 48  # 48 is proposed by the paper authors

        self.out_channels = out_channels
        self.output_stride = output_stride

        assert out_channels % group_size == 0
        self.aspp = nn.Sequential(
            ASPP(
                encoder_channels[-1],
                out_channels,
                atrous_rates,
                group_size,
                separable=True,
            ),
            SeparableConv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(),
        )

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        assert highres_out_channels % group_size == 0
        self.block1 = nn.Sequential(
            ScaledStdConv2d(
                highres_in_channels, highres_out_channels, kernel_size=1, bias=False
            ),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
        )

    def forward(self, *features):
        # Features is a list of six elements. Assuming the encoder is from timm
        # universal:
        #
        # The first is appended by segmentation_models_pytorch. From
        # 'segmentation_models_pytorch/encoders/timm_universal.py': 'forward()':
        # features[0] ([-6]): input
        #
        # The rest is from timm. The model is built with
        # 'build_model_with_cfg()' in 'timm/models/helpers.py'. The flag 'features_only'
        # is set to 'True' in 'segmentation_models_pytorch/encoders/timm_universal.py':
        # 'TimmUniversalEncoder()'. This means that features are collected in a
        # 'FeatureListNet' (from 'timm/models/features.py'). Which features are decided
        # by the elements in the 'feature_info' defined for each encoder class (e.g.
        # 'NormFreeNet()' in 'timm/models/nfnet.py'). For nfnet this is:
        # features[1] ([-5]): stem (64-channel, size / 2)
        # features[2] ([-4]): 1st nf stage (256-channel, size / 4)
        # features[3] ([-3]): 2nd nf stage (512-channel, size / 8)
        # features[4] ([-2]): 3rd nf stage (1536-channel, size / 16)
        # features[5] ([-1]): final conv expansion (3072-channel, size / 16)
        #
        # Note that the 4th nf stage feature is replaced by the final conv expansion
        # when this is applied (otherwise features[5] os the 4th nf stage). Also note
        # that the size / 16 dimension in the 4th nf stage and the expansion layer stems
        # from the deeplab decoder (see the 'output_stride' input to this class).

        # This is the same, but does not use less memory
        # return self.block2(
        #     torch.cat([
        #         self.up(self.aspp(features[-1])),
        #         self.block1(features[-4])
        #     ], dim=1)
        # )
        x = features[-1]
        y = features[-4]
        del features
        x = self.aspp(x)
        x = self.up(x)
        y = self.block1(y)
        x = torch.cat([x, y], dim=1)
        del y
        x = self.block2(x)
        return x


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, group_size):
        assert out_channels % group_size == 0
        super().__init__(
            ScaledStdConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.ReLU(),
        )


class ASPPSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, group_size):
        assert out_channels % group_size == 0
        super().__init__(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.ReLU(),
        )


class ASPP(nn.Module):
    def __init__(
        self, in_channels, out_channels, atrous_rates, group_size, separable=False
    ):
        assert out_channels % group_size == 0
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                ScaledStdConv2d(in_channels, out_channels, 1, bias=False),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1, group_size))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2, group_size))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3, group_size))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            ScaledStdConv2d(4 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class SeparableConv2d(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        dephtwise_conv = ScaledStdConv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = ScaledStdConv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)
