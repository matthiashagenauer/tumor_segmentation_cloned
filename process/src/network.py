import sys
import logging
from typing import Any

import torch
import segmentation_models_pytorch as smp

import configurations
import utils
import custom_fpn
from deeplabv3plus_gn.model import DeepLabV3PlusGN
from deeplabv3plus_nonorm.model import DeepLabV3PlusNoNorm
from deeplabv3plus_stdconv.model import DeepLabV3PlusStdConv

log = logging.getLogger("network")


def select(conf: configurations.Configurations) -> Any:
    """
    See https://github.com/qubvel/segmentation_models.pytorch for available encoders and
    decoders. Only some alternatives are enabled here.
    """
    if (
        conf.initialise_encoder is None
        or conf.initialise_encoder == "input"
        or not conf.train_mode
    ):
        encoder_weights = None
    else:
        encoder_weights = conf.initialise_encoder
    if conf.decoder == "fpn":
        if conf.encoder == "timm_nfnet_f7s":
            model = custom_fpn.FPN(
                encoder_name=conf.encoder,
                encoder_weights=encoder_weights,
                classes=len(conf.classes),
                activation=conf.activation,
            )
        else:
            model = smp.FPN(
                encoder_name=conf.encoder,
                encoder_weights=encoder_weights,
                classes=len(conf.classes),
                activation=conf.activation,
            )
    elif conf.decoder == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name=conf.encoder,
            encoder_depth=5,
            encoder_weights=encoder_weights,
            decoder_channels=256,
            in_channels=3,
            classes=len(conf.classes),
            activation=conf.activation,
            upsampling=8,
            aux_params=None,
        )
    elif conf.decoder == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=conf.encoder,
            encoder_depth=5,
            encoder_weights=encoder_weights,
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            in_channels=3,
            classes=len(conf.classes),
            activation=conf.activation,
            upsampling=4,
            aux_params=None,
        )
    elif conf.decoder == "deeplabv3plus_gn":
        model = DeepLabV3PlusGN(
            encoder_name=conf.encoder,
            encoder_depth=5,
            encoder_weights=encoder_weights,
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            in_channels=3,
            classes=len(conf.classes),
            activation=conf.activation,
            upsampling=4,
            aux_params=None,
        )
    elif conf.decoder == "deeplabv3plus_nonorm":
        model = DeepLabV3PlusNoNorm(
            encoder_name=conf.encoder,
            encoder_depth=5,
            encoder_weights=encoder_weights,
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            in_channels=3,
            classes=len(conf.classes),
            activation=conf.activation,
            upsampling=4,
            aux_params=None,
        )
    elif conf.decoder == "deeplabv3plus_stdconv":
        model = DeepLabV3PlusStdConv(
            encoder_name=conf.encoder,
            encoder_depth=5,
            encoder_weights=encoder_weights,
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            in_channels=3,
            classes=len(conf.classes),
            activation=conf.activation,
            upsampling=4,
            aux_params=None,
        )
    else:
        if conf.logger:
            log.error(f"Unimplemented decoder {conf.decoder}")
        sys.exit()

    if conf.initialise_encoder == "input" and conf.train_mode:
        log.info(f"Loading state from {conf.initialise_path}")
        state = torch.load(conf.initialise_path, map_location="cpu")
        net_state = utils.maybe_remove_module_prefix(state["network_state"])
        if conf.initialise_decoder:
            log.info("Initialising full model")
            model.load_state_dict(net_state)
        else:
            log.info("Initialising encoder")
            encoder_state = utils.extract_encoder(net_state)
            model.encoder.load_state_dict(encoder_state)

    return model
