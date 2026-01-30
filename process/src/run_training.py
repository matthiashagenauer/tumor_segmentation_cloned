import argparse
from pathlib import Path
import time
import traceback
import logging
from contextlib import suppress
from typing import Any, Dict, Mapping, Optional, Sequence

# Import numpy before pytorch in order to avoid error. The error is also present when
# numpy is not imported here. Related to https://github.com/pytorch/pytorch/issues/37377
import numpy as np
import torch
import timm
from timm.utils.agc import adaptive_clip_grad
from timm.utils.model_ema import ModelEmaV2
import segmentation_models_pytorch as smp

import configurations
import data
import network
import encoders_init
import utils
import step_length

log = logging.getLogger("train")


def get_current_step_length(optimiser: torch.optim.Optimizer) -> float:
    step_length = -1
    for param_group in optimiser.param_groups:
        if "lr" in param_group.keys():
            step_length = param_group["lr"]
            break
    return step_length


def get_optimiser(
    model: Any, conf: configurations.Configurations
) -> torch.optim.Optimizer:
    if "nfnet" in conf.encoder:
        # From (Brock, 2021) A1:
        #
        # > Critically, weight decay is not applied to the affine gains or biases in the
        # > weight-standardized convolutional layers, or to the SkipInit gains.
        #
        # We extend this to the decoder and segmentation head. Also we skip weight decay
        # regularisation on all biases (including fc biases). Finally, weight decay is
        # also skipped for group norm weights and biases in the FPN
        skip = []
        keep = []
        for n, p in model.named_parameters():
            if (
                n.endswith(".bias")
                or n.endswith(".gain")
                or n.endswith(".skipinit_gain")
                or len(p.shape) == 1
            ):
                skip.append(p)
            else:
                keep.append(p)
        parameters = [
            {"params": skip, "weight_decay": 0.0},
            {"params": keep, "weight_decay": conf.weight_decay},
        ]
        return torch.optim.SGD(
            params=parameters,
            lr=conf.initial_step_length,
            momentum=conf.momentum,
            nesterov=True,
            weight_decay=0.0,
        )
    else:
        return torch.optim.Adam(
            params=model.parameters(),
            lr=conf.initial_step_length,
        )


class TopKLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, k=10):
        self.k = k
        super(TopKLoss, self).__init__(reduction="none")

    def forward(self, prediction: torch.Tensor, reference: torch.Tensor):
        ce_result = super(TopKLoss, self).forward(prediction, reference)
        num_pixels = np.prod(ce_result.shape)
        result, _ = torch.topk(
            ce_result.view((-1,)), k=int(num_pixels * self.k / 100), sorted=False
        )
        return result.mean()


def get_loss_function(
    loss_function_str: Optional[str], device: str
) -> Optional[torch.nn.Module]:
    if loss_function_str is None:
        return None
    if loss_function_str == "cross_entropy":
        return torch.nn.CrossEntropyLoss().to(device)
    elif loss_function_str == "dice":
        return smp.losses.DiceLoss(mode="multiclass").to(device)
    elif loss_function_str == "ce_top10":
        return TopKLoss(k=10).to(device)
    elif loss_function_str == "ce_top50":
        return TopKLoss(k=50).to(device)
    elif loss_function_str == "ce_top90":
        return TopKLoss(k=90).to(device)
    else:
        log.error(f"Unimplemented loss function {loss_function_str}")
        return None


def log_progress(status: Mapping[str, Any]):
    log.info(
        "{:>8}{:>12,.3f}{:>12,.3f}{:>12,.3f}{:>12,.3f}{:>12,.5f}".format(
            status["step"],
            status["epoch"],
            status["loss"],
            status["images_per_sec"],
            status["sec_per_step"],
            status["step_length"],
        )
    )


def monitor_progress(
    filepath: Path,
    current_step: int,
    current_time: float,
    steps_since_last: int,
    start_step_time: float,
    steps_per_epoch: float,
    images_per_step: float,
    losses: Sequence[float],
    losses_1: Sequence[float],
    losses_2: Optional[Sequence[float]],
    step_length: float,
):
    sec_per_step = (current_time - start_step_time) / steps_since_last
    status = {
        "step": current_step,
        "epoch": current_step / steps_per_epoch,
        "loss": np.mean(losses),
        "images_per_sec": images_per_step / sec_per_step,
        "sec_per_step": sec_per_step,
        "step_length": step_length,
    }
    if losses_2 is not None:
        status["loss_1"] = np.mean(losses_1)
        status["loss_2"] = np.mean(losses_2)
    log_progress(status)
    utils.update_progress(status, filepath)


def maybe_state_dict(entity: Optional[Any]) -> Optional[Dict[str, Any]]:
    return None if entity is None else entity.state_dict()


def save_state(
    filepath: Path,
    network_state: Mapping[str, Any],
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    optimiser_state: Optional[Mapping[str, Any]] = None,
    schedule_1_state: Optional[Mapping[str, Any]] = None,
    schedule_2_state: Optional[Mapping[str, Any]] = None,
    network_state_ema: Optional[Mapping[str, Any]] = None,
    gradient_scaler_state: Optional[Mapping[str, Any]] = None,
):
    log.info(f"Writing state to {filepath}")
    state = {
        "epoch": epoch,
        "step": step,
        "network_state": network_state,
        "optimiser_state": optimiser_state,
    }
    if schedule_1_state is not None:
        state["schedule_1_state"] = schedule_1_state
    if schedule_2_state is not None:
        state["schedule_2_state"] = schedule_2_state
    if network_state_ema is not None:
        state["network_state_ema"] = network_state_ema
    if gradient_scaler_state is not None:
        state["gradient_scaler_state"] = gradient_scaler_state
    torch.save(state, filepath)


def monitor_input(
    out_dir: Path,
    image_paths: Sequence[Path],
    image_batch: torch.Tensor,
    reference_batch: torch.Tensor,
    logit_batch: torch.Tensor,
    class_labels: Sequence[int],
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = [out_dir.joinpath(Path(p).name) for p in image_paths]
    log.info(f"Writing batch to {out_dir}")

    images = image_batch.detach().cpu().numpy()
    references = reference_batch.detach().cpu().numpy()
    prediction_batch = utils.logit_to_prediction(logit_batch)
    predictions = np.floor(
        np.clip(
            255.0 * prediction_batch.detach().cpu().numpy(),
            0.0,
            255.0,
        )
    ).astype(np.uint8)
    utils.write_batch(
        out_paths,
        predictions,
        images=images,
        references=references,
        mean=mean,
        std=std,
        class_labels=class_labels,
    )


def monitor_performance(
    logit_batch: torch.Tensor,
    mask_batch: torch.Tensor,
    class_labels: Sequence[int],
):
    assert len(class_labels) > 1
    class_names = utils.add_class_names(class_labels)
    prediction_batch = utils.logit_to_prediction(logit_batch)
    predictions = prediction_batch.detach().cpu().numpy()
    prediction_masks = np.argmax(predictions, axis=1)
    reference_masks = mask_batch.cpu().numpy()
    if len(reference_masks.shape) == 3:
        class_indices = list(range(len(class_labels)))
        reference_label_encoding = True
        reference_values = set(np.unique(reference_masks))
        assert reference_values.difference(set(class_indices)) == set()
    else:
        reference_label_encoding = False
        assert len(reference_masks.shape) == 4
        assert reference_masks.shape[1] == len(class_labels)
    for class_label in class_labels:
        class_index = class_labels.index(class_label)
        class_name = class_names[class_label]
        log.info(f"Performance for class '{class_name}' vs. rest")
        if reference_label_encoding:
            binary_reference_masks = (reference_masks == class_index).astype(int)
        else:
            binary_reference_masks = reference_masks[:, class_index, :, :].astype(int)
        binary_prediction_masks = (prediction_masks == class_index).astype(int)

        metrics = utils.performance_evaluation(
            binary_prediction_masks, binary_reference_masks, False
        )
        for key, val in metrics.items():
            if val is not None:
                log.info(f"    {key:<30}: {val:>10,.4f}")
            else:
                val = "None"
                log.info(f"    {key:<30}: {val:>10}")


def train(conf: configurations.Configurations):
    torch.backends.cudnn.benchmark = True

    # Setup network
    net = network.select(conf)
    net.cuda()

    # Setup objective and optimiser
    optimiser = get_optimiser(net, conf)
    loss_function_1 = get_loss_function(conf.loss_function_1, conf.device)
    if loss_function_1 is None:
        raise ValueError("Loss function 1 is None")
    loss_function_2 = get_loss_function(conf.loss_function_2, conf.device)
    schedule_1 = step_length.get_schedule(
        optimiser, conf.schedule_1, conf.schedule_1_config
    )
    schedule_2 = step_length.get_schedule(
        optimiser, conf.schedule_2, conf.schedule_2_config
    )

    # Setup automatic mixed precision (AMP)
    amp_autocast = torch.cuda.amp.autocast if conf.amp else suppress
    gradient_scaler = torch.cuda.amp.GradScaler() if conf.amp else None

    # Setup exponential moving average (EMA) tracking of network parameters
    net_ema = None
    if conf.ema:
        # From pytorch_image_models/train.py:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before
        # SyncBN and DDP wrapper
        # NOTE: We use "warm up" decay rate to mitigate zero bias because of zero
        # initialisation (see TF1 implementation of ExponentialMovingAverage)
        #     actual_decay = min{decay, (1 + step) / (10 + step)}
        # This warmup is implemented in the update below in the optimisation loop
        net_ema = ModelEmaV2(net, decay=0.9999, device="cpu")

    # Setup distributed optimisation
    if conf.distributed:
        if conf.logger:
            log.info(f"Distributed training with world size {conf.world_size}")
        # Manuellay adjust batch_size and loading workers
        batch_size = int(conf.batch_size / conf.num_gpus)
        num_workers = int((conf.num_workers + conf.num_gpus - 1) / conf.num_gpus)
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[conf.local_rank],
            # find_unused_parameters=True,
        )
    else:
        batch_size = conf.batch_size
        num_workers = conf.num_workers
        if conf.num_gpus > 1:
            if conf.logger:
                log.info(f"Parallel training with {conf.num_gpus} gpus")
            net = torch.nn.DataParallel(net)
        else:
            if conf.logger:
                log.info(f"Single-device training on {conf.device}")

    if conf.restore_path is not None:
        state = torch.load(conf.restore_path, map_location="cpu")
        net.load_state_dict(utils.maybe_remove_module_prefix(state["network_state"]))
        optimiser.load_state_dict(state["optimiser_state"])
        if schedule_1 is not None:
            schedule_1.load_state_dict(state["schedule_1_state"])
        if schedule_2 is not None:
            schedule_2.load_state_dict(state["schedule_2_state"])
        if gradient_scaler is not None:
            gradient_scaler.load_state_dict(state["gradient_scaler_state"])
        if conf.ema:
            assert net_ema is not None
            if "network_state_ema" in state.keys():
                net_ema.module.load_state_dict(state["network_state_ema"])
            else:
                log.warning(
                    "Tried to load exponential moving average model but it does not \n"
                    "exist in stored checkpoint. Try to restore non-ema model instead."
                )
                net_ema.module.load_state_dict(state["network_state"])
        epoch = state["epoch"]
        step = state["step"]
        net = net.to(conf.device)
    else:
        epoch = 0
        step = 0

    net.train()
    utils.network_summary(
        conf.log_dir,
        net,
        conf.logger,
        conf.encoder,
        conf.distributed,
        [batch_size, 3, conf.target_height, conf.target_width],
    )

    # Load data
    initialised_preprocessing = (
        None if conf.initialise_encoder in (None, "input") else conf.initialise_encoder
    )
    preprocessing_fn = encoders_init.get_preprocessing_fn(
        conf.encoder,
        initialised_preprocessing,
        conf.input_space,
        conf.input_range,
        conf.train_mean,
        conf.train_std,
    )
    dataset_df = data.get_data_paths(conf.input_data_path)
    dataset = data.SegmentedImages(
        dataset_df,
        conf.classes,
        conf.train_mode,
        conf.min_divisor,
        data.get_training_transform(conf),
        preprocessing_fn,
    )
    if conf.distributed:
        train_sampler: Optional[
            torch.utils.data.distributed.DistributedSampler
        ] = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    else:
        train_sampler = None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        drop_last=conf.drop_last_batch,
        pin_memory=True,
        sampler=train_sampler,
    )

    # Log setup
    dataset_size = len(dataset)
    if conf.drop_last_batch:
        images_per_epoch = dataset_size // conf.batch_size * conf.batch_size
    else:
        images_per_epoch = dataset_size
    images_per_step = conf.batch_size
    steps_per_epoch = images_per_epoch / images_per_step
    num_epochs = conf.max_steps / steps_per_epoch
    if conf.logger:
        log.info("Start training")
        log.info(f"Number of gpus:     {conf.num_gpus:>7}")
        log.info(f"Dataset size:       {dataset_size:>7}")
        log.info(f"Examples per epoch: {images_per_epoch:>7}")
        log.info(f"Examples per step:  {images_per_step:>7}")
        log.info(f"Steps per epoch:    {steps_per_epoch:>10,.2f}")
        log.info(f"Number of steps:    {conf.max_steps:>7}")
        log.info(f"Number of epochs:   {num_epochs:>10,.2f}")

    # Prepare optimisation start
    steps_since_last = 0
    start_step_time = time.time()
    losses = []
    losses_1 = []
    if loss_function_2 is None:
        losses_2 = None
    else:
        losses_2 = []
    progress_header = "{:>8}{:>12}{:>12}{:>12}{:>12}{:>12}".format(
        "Step",
        "Epoch",
        "Loss",
        "Images/sec",
        "Secs/step",
        "Step length",
    )

    # Start optimisation. Outer loop is over "epochs", inner loop is over dataset
    while True:
        if conf.logger:
            # Print header to make it easier to follow on the monitor
            log.info(progress_header)
        for sample_batch in dataloader:
            # Load batch
            image_batch = sample_batch["image"].to(conf.device)
            mask_batch = sample_batch["mask"].to(conf.device)

            # Perform step
            with amp_autocast():
                # NOTE: suppressed if not amp
                logit_batch = net.forward(image_batch)
                if conf.amp:
                    logit_batch = logit_batch.type(torch.float32).to(conf.device)
                loss_1 = loss_function_1.forward(logit_batch, mask_batch)
                if loss_function_2 is None:
                    loss = loss_1
                else:
                    loss_2 = loss_function_2.forward(logit_batch, mask_batch)
                    loss_value_2 = loss_2.data
                    loss = loss_1 + loss_2

            loss_value_1 = loss_1.data
            loss_value = loss.data
            if conf.distributed:
                loss_value_1 = utils.tensor_reduce_average(
                    loss_value_1, conf.world_size
                )
                if loss_function_2 is not None:
                    loss_value_2 = utils.tensor_reduce_average(
                        loss_value_2, conf.world_size
                    )
                loss_value = utils.tensor_reduce_average(loss_value, conf.world_size)

            if torch.isnan(loss_value).any():
                # TODO: Write input and result for all processes. Require existing write
                # dir for all
                if conf.logger:
                    log.error("Encountered nan in loss")
                for p in net.parameters():
                    if torch.isnan(p).any():
                        log.error(f"Rank {conf.local_rank}: Parameter {p}")
                for p in net.parameters():
                    if torch.isnan(p.grad).any():
                        log.error(f"Rank {conf.local_rank}: Gradient {p.grad}")
                log.error(
                    f"Input: {sample_batch['image_path']}, {sample_batch['mask_path']}"
                )
                if conf.logger:
                    save_state(
                        conf.checkpoint_dir.joinpath(f"model-state_step-{step:06}.tar"),
                        net.state_dict(),
                        step,
                        epoch,
                        optimiser.state_dict(),
                        maybe_state_dict(schedule_1),
                        maybe_state_dict(schedule_2),
                        maybe_state_dict(net_ema),
                        maybe_state_dict(gradient_scaler),
                    )
                return

            optimiser.zero_grad()

            if conf.amp:
                assert gradient_scaler is not None
                gradient_scaler.scale(loss).backward()
                if conf.agc:
                    gradient_scaler.unscale_(optimiser)
                    adaptive_clip_grad(
                        net.parameters(), clip_factor=conf.agc_factor, eps=1e-3
                    )
                gradient_scaler.step(optimiser)
                gradient_scaler.update()
            else:
                loss.backward()
                if conf.agc:
                    adaptive_clip_grad(
                        net.parameters(), clip_factor=conf.agc_factor, eps=1e-3
                    )
                optimiser.step()

            if conf.ema:
                assert net_ema is not None
                decay = min(net_ema.decay, (1 + step) / (10 + step))
                net_ema._update(net, lambda e, m: decay * e + (1.0 - decay) * m)
            torch.cuda.synchronize()
            step += 1

            # Update step length schedule
            if schedule_1 is not None and step < conf.switch_schedule:
                schedule_1.step()
            if schedule_2 is not None and step >= conf.switch_schedule:
                schedule_2.step()

            # Monitoring
            if conf.logger:
                steps_since_last += 1
                losses.append(loss_value.item())
                losses_1.append(loss_value_1.item())
                if loss_function_2 is not None:
                    losses_2.append(loss_value_2.item())

                if step % conf.monitor_progress == 0 or step <= 10:
                    monitor_progress(
                        conf.log_dir.joinpath("progress.csv"),
                        step,
                        time.time(),
                        steps_since_last,
                        start_step_time,
                        steps_per_epoch,
                        images_per_step,
                        losses,
                        losses_1,
                        losses_2,
                        get_current_step_length(optimiser),
                    )
                    steps_since_last = 0
                    start_step_time = time.time()
                    losses = []
                    losses_1 = []
                    losses_2 = None if losses_2 is None else []

                if step == conf.max_steps:
                    save_state(
                        conf.checkpoint_dir.joinpath(f"model-state_step-{step:06}.tar"),
                        net.state_dict(),
                        step,
                        epoch,
                        optimiser.state_dict(),
                        maybe_state_dict(schedule_1),
                        maybe_state_dict(schedule_2),
                        maybe_state_dict(net_ema),
                        maybe_state_dict(gradient_scaler),
                    )
                if step % conf.periodic_checkpoint == 0 and step != conf.max_steps:
                    save_state(
                        conf.checkpoint_dir.joinpath(f"model-state_step-{step:06}.pth"),
                        net.state_dict(),
                    )

            gathered = False
            if step % conf.monitor_input == 0:
                image_paths = sample_batch["image_path"]
                if conf.distributed:
                    # NOTE: pytorch all_gather does not work on python lists of paths,
                    # so this is a workaround that lets us write all input images, but
                    # we lose the origin.
                    image_paths = [Path("image.png")] * conf.batch_size
                    image_batch = utils.tensor_all_gather(image_batch, conf.num_gpus)
                    mask_batch = utils.tensor_all_gather(mask_batch, conf.num_gpus)
                    logit_batch = utils.tensor_all_gather(logit_batch, conf.num_gpus)
                    gathered = True
                if conf.logger:
                    overwrite = True
                    out_dir = conf.debug_dir.joinpath("images")
                    out_dir = out_dir if overwrite else out_dir.joinpath(f"s-{step:06}")
                    monitor_input(
                        out_dir,
                        image_paths,
                        image_batch,
                        mask_batch,
                        logit_batch,
                        class_labels=conf.classes,
                        mean=conf.train_mean,
                        std=conf.train_std,
                    )

            if step % conf.monitor_performance == 0:
                if conf.distributed and not gathered:
                    logit_batch = utils.tensor_all_gather(logit_batch, conf.num_gpus)
                    mask_batch = utils.tensor_all_gather(mask_batch, conf.num_gpus)
                if conf.logger:
                    monitor_performance(logit_batch, mask_batch, conf.classes)
                    log.info(progress_header)

            # Check termination criterion
            if step >= conf.max_steps:
                return
            # Completed step
        if conf.distributed:
            timm.utils.distribute_bn(net, conf.world_size, reduce=True)
        # Completed epoch


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    conf = configurations.set_up(parser, True)
    if conf.logger:
        log.info("Configurations is set up, start program")

    try:
        train(conf)
    except Exception as err:
        log.error(f"Program finished with exception: \n{err}")
        log.error(traceback.format_exc())
    else:
        if conf.logger:
            log.info("Program finished correctly")
    finally:
        if conf.logger:
            formatted = utils.format_time(time.time() - start_time)
            log.info(f"Elapsed time: {formatted}")


if __name__ == "__main__":
    main()
