import logging
from typing import Any, Dict, List, Sequence, Mapping

import torch
from segmentation_models_pytorch.encoders._base import EncoderMixin
from timm.models.nfnet import NormFreeNet, default_cfgs, model_cfgs, NfCfg

log = logging.getLogger("nfnet")


class NormFreeNetEncoder(NormFreeNet, EncoderMixin):
    def __init__(self, model_cfg: Any, depth: int = 5, **kwargs):
        super().__init__(model_cfg, **kwargs)

        self._depths = model_cfg.depths
        self._out_channels = tuple(
            [3] + list(model_cfg.channels)[:-1] + [self.num_features]
        )
        self._depth = depth
        self._in_channels = 3

        del self.head

    def get_stages(self) -> List[torch.nn.Module]:
        feature_modules = [f["module"] for f in self.feature_info]

        # Append first 3 / 4 of stem in the first, the last in the second
        first_stem = []
        second_stem = []
        first = True
        for n, m in self.stem.named_modules():
            if isinstance(m, torch.nn.Sequential):
                continue
            if first:
                first_stem.append(m)
            else:
                second_stem.append(m)
            if f"stem.{n}" in feature_modules:
                first = False

        stages: List[torch.nn.Module] = [torch.nn.Identity()]
        stages.append(torch.nn.Sequential(*first_stem))
        stages.append(torch.nn.Sequential(*second_stem, *self.stages[0]))
        stages.append(self.stages[1])
        stages.append(self.stages[2])
        stages.append(torch.nn.Sequential(*self.stages[3], self.final_conv))

        return stages

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        stages = self.get_stages()

        features: List[torch.Tensor] = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            # Originally, all features and input were stored. This is waste memory in
            # run time since decoders rarly use all features.
            # The below is hardcoded to only keep features[-4] and features[-1] which
            # are used in deeplab v3+ decoders.
            # TODO: Make flexible so that it works for other decoders also

            if i in [2, self._depth]:
                features.append(x)
            else:
                features.append(torch.tensor([0]))

        return features

    def load_state_dict(self, state_dict: Dict[str, Any], **kwargs):
        if "head.fc.bias" in state_dict.keys():
            state_dict.pop("head.fc.bias")
        if "head.fc.weight" in state_dict.keys():
            state_dict.pop("head.fc.weight")
        super().load_state_dict(state_dict, **kwargs)

    # def make_dilated(self, stage_list, dilation_list):
    #     raise ValueError("NFNet encoders do not implement dilated mode")


def prepare_settings(settings: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "mean": settings["mean"],
        "std": settings["std"],
        "url": settings["url"],
        "input_range": (0, 1),
        "input_space": "RGB",
    }


def _nfnet_cfg(
    depths,
    channels=(256, 512, 1536, 1536),
    group_size=128,
    bottle_ratio=0.5,
    feat_mult=2.0,
    act_layer="gelu",
    attn_layer="se",
    attn_kwargs=dict(rd_ratio=0.5),
):
    cfg = NfCfg(
        depths=depths,
        channels=channels,
        stem_type="deep_quad",
        stem_chs=128,
        group_size=group_size,
        bottle_ratio=bottle_ratio,
        extra_conv=True,
        num_features=int(channels[-1] * feat_mult),
        act_layer=act_layer,
        attn_layer=attn_layer,
        attn_kwargs=attn_kwargs,
    )
    return cfg


def _dm_nfnet_cfg(
    depths,
    channels=(256, 512, 1536, 1536),
    act_layer="gelu",
    skipinit=True,
    attn_layer="se",
    attn_kwargs=dict(rd_ratio=0.5),
):
    cfg = NfCfg(
        depths=depths,
        channels=channels,
        stem_type="deep_quad",
        stem_chs=128,
        group_size=128,
        bottle_ratio=0.5,
        extra_conv=True,
        gamma_in_act=True,
        same_padding=True,
        skipinit=skipinit,
        num_features=int(channels[-1] * 2.0),
        act_layer=act_layer,
        attn_layer=attn_layer,
        attn_kwargs=attn_kwargs,
    )
    return cfg


# Add configs without attention
model_cfgs.update(
    dict(
        # NFNet-F models w/ GELU compatible with DeepMind weights
        dm_nfnet_noattn_f0=_dm_nfnet_cfg(depths=(1, 2, 6, 3), attn_layer=None),
        dm_nfnet_noattn_f1=_dm_nfnet_cfg(depths=(2, 4, 12, 6), attn_layer=None),
        dm_nfnet_noattn_f2=_dm_nfnet_cfg(depths=(3, 6, 18, 9), attn_layer=None),
        dm_nfnet_noattn_f3=_dm_nfnet_cfg(depths=(4, 8, 24, 12), attn_layer=None),
        dm_nfnet_noattn_f4=_dm_nfnet_cfg(depths=(5, 10, 30, 15), attn_layer=None),
        dm_nfnet_noattn_f5=_dm_nfnet_cfg(depths=(6, 12, 36, 18), attn_layer=None),
        dm_nfnet_noattn_f6=_dm_nfnet_cfg(depths=(7, 14, 42, 21), attn_layer=None),
        # NFNet-F models w/ SiLU (much faster in PyTorch)
        nfnet_noattn_f0=_nfnet_cfg(depths=(1, 2, 6, 3), attn_layer=None),
        nfnet_noattn_f1=_nfnet_cfg(depths=(2, 4, 12, 6), attn_layer=None),
        nfnet_noattn_f2=_nfnet_cfg(depths=(3, 6, 18, 9), attn_layer=None),
        nfnet_noattn_f3=_nfnet_cfg(depths=(4, 8, 24, 12), attn_layer=None),
        nfnet_noattn_f4=_nfnet_cfg(depths=(5, 10, 30, 15), attn_layer=None),
        nfnet_noattn_f5=_nfnet_cfg(depths=(6, 12, 36, 18), attn_layer=None),
        nfnet_noattn_f6=_nfnet_cfg(depths=(7, 14, 42, 21), attn_layer=None),
        nfnet_noattn_f7=_nfnet_cfg(depths=(8, 16, 48, 24), attn_layer=None),
        # Experimental 'light' versions of NFNet-F that are little leaner
        nfnet_noattn_l0=_nfnet_cfg(
            depths=(1, 2, 6, 3),
            feat_mult=1.5,
            group_size=64,
            bottle_ratio=0.25,
            attn_layer=None,
            attn_kwargs=dict(),
            act_layer="silu",
        ),
        nfnet_noattn_l1=_nfnet_cfg(
            depths=(2, 4, 12, 6),
            feat_mult=2,
            group_size=64,
            bottle_ratio=0.25,
            attn_layer=None,
            attn_kwargs=dict(),
            act_layer="silu",
        ),
        nfnet_noattn_l2=_nfnet_cfg(
            depths=(3, 6, 18, 9),
            feat_mult=2,
            group_size=64,
            bottle_ratio=0.25,
            attn_layer=None,
            attn_kwargs=dict(),
            act_layer="silu",
        ),
        nfnet_noattn_l3=_nfnet_cfg(
            depths=(4, 8, 24, 12),
            feat_mult=2,
            group_size=64,
            bottle_ratio=0.25,
            attn_layer=None,
            attn_kwargs=dict(),
            act_layer="silu",
        ),
    )
)

# Description comments are from timm/models/nfnet.py
# Drop rates are inferred from Table 1 in https://arxiv.org/abs/2102.06171 (Brock, 2021)
encoder_params: Sequence[Dict[str, Any]] = [
    {"name": "dm_nfnet_f0", "drop_rate": 0.2},
    {"name": "dm_nfnet_f1", "drop_rate": 0.3},
    {"name": "dm_nfnet_f2", "drop_rate": 0.4},
    {"name": "dm_nfnet_f3", "drop_rate": 0.4},
    {"name": "dm_nfnet_f4", "drop_rate": 0.5},
    {"name": "dm_nfnet_f5", "drop_rate": 0.5},
    {"name": "dm_nfnet_f6", "drop_rate": 0.5},
    {"name": "nfnet_f0s", "drop_rate": 0.2},
    {"name": "nfnet_f1s", "drop_rate": 0.3},
    {"name": "nfnet_f2s", "drop_rate": 0.4},
    {"name": "nfnet_f3s", "drop_rate": 0.4},
    {"name": "nfnet_f4s", "drop_rate": 0.5},
    {"name": "nfnet_f5s", "drop_rate": 0.5},
    {"name": "nfnet_f6s", "drop_rate": 0.5},
    {"name": "nfnet_f7s", "drop_rate": 0.5},
    {"name": "dm_nfnet_noattn_f0", "drop_rate": 0.2},
    {"name": "dm_nfnet_noattn_f1", "drop_rate": 0.3},
    {"name": "dm_nfnet_noattn_f2", "drop_rate": 0.4},
    {"name": "dm_nfnet_noattn_f3", "drop_rate": 0.4},
    {"name": "dm_nfnet_noattn_f4", "drop_rate": 0.5},
    {"name": "dm_nfnet_noattn_f5", "drop_rate": 0.5},
    {"name": "dm_nfnet_noattn_f6", "drop_rate": 0.5},
    {"name": "nfnet_noattn_f0", "drop_rate": 0.2},
    {"name": "nfnet_noattn_f1", "drop_rate": 0.3},
    {"name": "nfnet_noattn_f2", "drop_rate": 0.4},
    {"name": "nfnet_noattn_f3", "drop_rate": 0.4},
    {"name": "nfnet_noattn_f4", "drop_rate": 0.5},
    {"name": "nfnet_noattn_f5", "drop_rate": 0.5},
    {"name": "nfnet_noattn_f6", "drop_rate": 0.5},
    {"name": "nfnet_noattn_f7", "drop_rate": 0.5},
    {"name": "eca_nfnet_l0", "drop_rate": 0.2},
    {"name": "eca_nfnet_l1", "drop_rate": 0.3},
    {"name": "eca_nfnet_l2", "drop_rate": 0.4},
    {"name": "eca_nfnet_l3", "drop_rate": 0.4},
    {"name": "nfnet_noattn_l0", "drop_rate": 0.2},
    {"name": "nfnet_noattn_l1", "drop_rate": 0.3},
    {"name": "nfnet_noattn_l2", "drop_rate": 0.4},
    {"name": "nfnet_noattn_l3", "drop_rate": 0.4},
]

timm_nfnet_encoders = {
    f"timm-{param['name'].replace('_', '-')}": {
        "encoder": NormFreeNetEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(
                default_cfgs[
                    "eca_" + param["name"].replace("_noattn", "")
                    if param["name"].startswith("nfnet_noattn_l")
                    else param["name"].replace("_noattn", "")
                ]
            )
        },
        "params": {
            "model_cfg": model_cfgs[param["name"]],
            "drop_rate": param["drop_rate"],
        },
    }
    for param in encoder_params
}
