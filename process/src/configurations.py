import sys
import os
import shutil
import logging
import logging.config
import datetime
from pathlib import Path
from collections import OrderedDict
from typing import Any, Optional, MutableMapping, Dict
import argparse

import numpy as np
import torch
import toml  # type: ignore

import utils


log: logging.Logger = logging.getLogger()


class Configurations:
    __initialised = False

    def __init__(self):
        """
        Default values are set here, and all attributions need to be initialised in this
        method.

        The values can be overrided by command line input or the input config file.

        If no input config file is given when restoring a training session, the config
        file of the training to be resumed is tried before falling back to default
        configs.
        """
        # GENERAL
        self.train_mode = True
        self.verbose = 1
        self.gpu = None
        self.rng_seed = None
        self.config_name = "default"
        self.inference_time = None
        self.restore_step = None
        self.start_time = None
        self.overwrite = None

        # GPUs
        self.device = "cuda:0"
        self.distribution_backend = "nccl"
        self.world_size = None
        self.local_rank = None
        self.rank = 0
        self.distributed = False
        self.logger = False

        # PATHS
        self.src_dir = None
        self.debug_dir = None
        self.checkpoint_dir = None
        self.log_dir = None
        self.log_src_dir = None
        self.log_file = None
        self.result_dir = None
        self.probability_map_dir = None
        self.restore_path = None
        self.initialise_path = None
        self.input_data_path = None
        self.output_dir = None
        self.run_dir = None
        self.config_file = None
        self.full_image_path = None

        # DATA
        self.batch_size = 16
        self.target_height = 1376
        self.target_width = 1376
        self.classes = [0, 255]

        # RUN
        self.num_gpus = 8
        self.num_workers = 32
        self.max_steps = 200000
        self.drop_last_batch = True
        self.inference_per_scan = False
        self.write_features = False
        # Restore timm-universal (tu) model and use it in non-timm-universal network
        self.restore_universal_to_non_universal = False

        # INFERENCE POSTPROCESS
        # Only if inference_per_scan and merge_tile_predictions. Apply the following on
        # the merged (and possibly downscaled) images
        # - Resize tiles with resize_factor (if --full_image is given)
        # - Smooth tiles
        # - Align tiles
        # - Merge tiles
        # - Resize merged image to corresponding full image (if --full_image is given)
        # - Clean result and segment into mask image with --classes values
        self.merge_tile_predictions = False
        # Only relevant if merge_tile_predictions=True
        self.align_tiles = False
        self.write_tile_predictions = True
        self.segment_merged_prediction = False
        self.postprocess_smooth = False
        self.postprocess_method = "argmax"
        self.postprocess_prune = None
        self.postprocess_close_holes = False
        self.segmentation_name = None  # Automatically generated from settings

        # NETWORK
        self.encoder = "tu-dm_nfnet_f6"
        self.decoder = "fpn"
        # could be None for logits or 'softmax2d' for multicalss segmentation
        self.activation = None
        # Minimal number inference image height and width must be divisible by
        self.min_divisor = 16

        # DATA
        # Either initialise_encoder is not None or the rest must be given
        # initialise encoder is
        #    None: training from scratch
        #    input: using model from --initialise cli argument
        #    imagenet: using pretrained model on imagenet if it exists
        # initialise decoder is
        #    False: No
        #    True: using model from --initialise cli argument
        self.initialise_encoder = None
        self.initialise_decoder = False
        self.input_space = "RGB"
        self.input_range = [0, 1]
        self.distort_basic = True  # Flipping and rotation
        self.distort_morphology = False  # Stretching, gauss noise, etc
        self.distort_bc = False  # Brightness and contrast
        self.distort_hsv = False  # Shift hue saturation and value
        # RGB
        self.train_mean = [8.297992e-01, 7.106879e-01, 8.241846e-01]
        self.train_std = [1.051075e-01, 1.543867e-01, 9.917571e-02]

        # OPTIMISATION
        self.momentum = 0.9
        self.weight_decay = 2e-5
        self.initial_step_length = 1.0e-5
        self.loss_function_1 = "cross_entropy"
        self.loss_function_2 = None
        self.schedule_1 = None
        self.schedule_1_config = {}
        self.schedule_2 = None
        self.schedule_2_config = {}
        self.switch_schedule = 10000
        self.agc = True  # Adaptive Gradient Clipping
        self.amp = True  # Automatic Mixed Precision in training and inference
        self.ema = False  # Exponential Moving Average
        self.agc_factor = 0.01

        # MONITOR
        self.monitor_progress = 20
        self.monitor_performance = 200
        self.monitor_input = 1000000
        self.periodic_checkpoint = 5000

        self.__initialised = True

    def __setattr__(self, name: str, value: Any):
        """
        Custom assignment method to ensure that all attributes we are trying to set are
        declared and initialised in __init__.
        """
        if self.__initialised and not hasattr(self, name):
            raise AttributeError(f"Trying to set uninitialised attribute: {name}")
        else:
            super().__setattr__(name, value)

    def update(self, config_dict: MutableMapping[str, Any]):
        unexpected = []
        for k in config_dict.keys():
            if k not in vars(self).keys():
                unexpected.append(k)
        if unexpected != []:
            print("ERROR: Unexpected config key(s) found in command line input")
            for key in unexpected:
                print(f"{key}")
            sys.exit()

        for k, v in config_dict.items():
            if v is not None:
                setattr(self, k, v)


def update_config(
    config: Configurations,
    from_file: MutableMapping[str, Any],
    from_cli: MutableMapping[str, Any],
) -> Configurations:
    # Update config from input config file
    # for k, v in from_file.items():
    #     # Convert arrays to Range
    #     if k.startswith("scale") or k.startswith("shift"):
    #         assert len(v) == 2, f"Invalid length for input config {k}: {v}"
    #         from_file[k] = Range(v[0], v[1])
    config.update(from_file)
    config.update(from_cli)

    return config


def parse_cli(argparser: argparse.ArgumentParser, train_mode: bool) -> Dict[str, Any]:
    argparser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=3,
        metavar="INT",
        help="Verbosity level [3]",
    )
    argparser.add_argument(
        "-c",
        "--config",
        metavar="PATH",
        help="Config file (.toml)",
    )
    argparser.add_argument(
        "-i",
        "--input",
        metavar="PATH",
        required=True,
        help="Csv file with one data example per record.\n"
        "For training, its header and the first record is on the form\n"
        "\n"
        "ImagePath,MaskPath\n"
        "/path/to/image.extension,/path/to/corresponding_mask.extension\n"
        "\n"
        "When applying a model (inference), masks are optional. If masks are\n"
        "included they enable on-the-fly evaluation using proxy metrics",
    )
    argparser.add_argument(
        "-r",
        "--restore",
        metavar="PATH",
        help="Filepath to checkpoint used to restore a model",
    )
    argparser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Root path to place output. Required if --restore is not specified,\n"
        "otherwise it can be inferred.",
    )
    argparser.add_argument(
        "-n",
        "--name",
        metavar="STR",
        help="Configuration name",
    )
    argparser.add_argument(
        "-g",
        "--gpu",
        metavar="INT",
        type=int,
        help="Which gpu to use when applying a segmentation model",
    )
    if train_mode:
        argparser.add_argument(
            "--initialise",
            metavar="PATH",
            help="Filepath to checkpoint used to initialise a model",
        )
    else:
        argparser.add_argument(
            "--full_image",
            metavar="PATH",
            help="Path to full images that correspond to input tiles. When\n"
            "`merge_tile_predictions` is `True`, resize the merged image to the\n"
            "size of the corresponding full image",
        )
        argparser.add_argument(
            "--run_dir",
            metavar="PATH",
            help=(
                "Set run dir directly. Overrides inferred output from --output and "
                "--restore"
            )
        )
        argparser.add_argument(
            "--overwrite",
            metavar="STR",
            choices=["y", "yes", "n", "no"],
            help="Decide what to do if resulting probability maps exist.",
        )
    args = argparser.parse_args()

    def map_some(option_value, function):
        return None if option_value is None else function(option_value)

    conf = {}
    conf["verbose"] = args.verbose
    conf["config_file"] = map_some(args.config, Path)
    conf["restore_path"] = map_some(args.restore, Path)
    conf["input_data_path"] = map_some(args.input, Path)
    conf["output_dir"] = map_some(args.output, Path)
    conf["config_name"] = args.name
    conf["gpu"] = args.gpu
    if train_mode:
        conf["initialise_path"] = map_some(args.initialise, Path)
    else:
        conf["overwrite"] = args.overwrite
        conf["full_image_path"] = map_some(args.full_image, Path)
        conf["run_dir"] = map_some(args.run_dir, Path)
    return conf


def print_command_line_args(conf: Configurations):
    log.info("Command line arguments used to run the program:")
    log.info(f"--verbose:     {conf.verbose}")
    log.info(f"--config:      {conf.config_file}")
    log.info(f"--input:       {conf.input_data_path}")
    log.info(f"--restore:     {conf.restore_path}")
    log.info(f"--output:      {conf.output_dir}")
    log.info(f"--name:        {conf.config_name}")
    log.info(f"--gpu:         {conf.gpu}")
    if conf.train_mode:
        log.info(f"--initialise:  {conf.initialise_path}")
    else:
        log.info(f"--run_dir:     {conf.run_dir}")
        log.info(f"--full_image:  {conf.full_image_path}")
        log.info(f"--overwrite:   {conf.overwrite}")


def extract_logged_config(restore_path: Path) -> Dict[str, Any]:
    config_path = restore_path.parents[1].joinpath(
        "logs", "network_training_config.toml"
    )
    with config_path.open("r") as ifile:
        all_configs = toml.loads(ifile.read())
    # Inherit all configs except train_mode, start_time, restore_step, and paths
    keys_to_exclude = [
        "train_mode",
        "start_time",
        "restore_step",
        "src_dir",
        "debug_dir",
        "checkpoint_dir",
        "log_dir",
        "log_src_dir",
        "log_file",
        "result_dir",
        "probability_map_dir",
        "restore_path",
        "initialise_path",
        "input_data_path",
        "output_dir",
        "run_dir",
        "config_file",
        "full_image_path",
    ]
    config = {k: v for k, v in all_configs.items() if k not in keys_to_exclude}
    return config


def check_input_config(conf: Configurations, train_mode: bool):
    if not train_mode and conf.restore_path is None:
        print("ERROR: --restore is not set")
        sys.exit()

    if conf.restore_path is None and conf.output_dir is None:
        print("ERROR: --output is not set")
        sys.exit()

    if conf.restore_path is not None and conf.initialise_path is not None:
        print("ERROR: Both --restore and --initialise is set")
        sys.exit()

    if conf.initialise_encoder == "input" and conf.initialise_path is None:
        print("ERROR: --initialise is not set, but initialise_encode = 'input'")
        sys.exit()

    if conf.initialise_decoder and conf.initialise_path is None:
        print("ERROR: --initialise is not set, but initialise_decoder = True")
        sys.exit()

    if conf.initialise_path is not None and conf.initialise_encoder != "input":
        print("ERROR: --initialise is set but initialise_encoder != 'input'")
        sys.exit()


def set_up(argparser: argparse.ArgumentParser, train_mode: bool) -> Configurations:
    """
    Set up configurations according to the default configurations, and the possibly
    updated config from a config file.
    """

    config_from_cli = parse_cli(argparser, train_mode)

    if config_from_cli["config_file"] is not None:
        with config_from_cli["config_file"].open("r") as ifile:
            config_from_file = toml.loads(ifile.read())
    else:
        if train_mode and config_from_cli["restore_path"] is not None:
            config_from_file = extract_logged_config(config_from_cli["restore_path"])
        else:
            config_from_file = {}

    if config_from_cli["config_name"] is None:
        _ = config_from_cli.pop("config_name")

    config = Configurations()
    config = update_config(config, config_from_file, config_from_cli)
    check_input_config(config, train_mode)

    # Set up program
    config = additional_setup(config, train_mode)
    if train_mode and config_from_cli["restore_path"] is not None:
        # Print here as logging is created in additional_setup
        if config.logger:
            log.info("Inherit config from restored model")
    if config.logger:
        print_command_line_args(config)

    return config


def additional_setup(conf: Configurations, train_mode: bool) -> Configurations:
    """Automatic setup based on the configurations above."""

    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=1)))
    conf.start_time = (
        f"{now.year:04}-{now.month:02}-{now.day:02}_"
        f"{now.hour:02}-{now.minute:02}-{now.second:02}"
    )
    conf.train_mode = train_mode
    conf.src_dir = Path(__file__).resolve().parent

    if conf.train_mode:
        assert torch.cuda.is_available(), "CUDA is unavailable"
        try:
            conf.local_rank = int(os.environ["LOCAL_RANK"])
        except KeyError:
            if conf.gpu:
                if conf.num_gpus == 0:
                    print("ERROR: Expected num_gpus to be > 0")
                    sys.exit()
                elif conf.num_gpus == 1:
                    conf.local_rank = 0
                else:
                    print("ERROR: No env variable 'LOCAL_RANK' in multi-gpu run")
                    sys.exit()
            else:
                conf.local_rank = None
        except Exception as e:
            print("WARNING: Setting local rank to None:", e)
            conf.local_rank = None
        conf.num_gpus = min(conf.num_gpus, torch.cuda.device_count())
    else:
        conf.num_gpus = 0 if conf.gpu is None else 1
        conf.inference_time = conf.start_time
        if conf.segment_merged_prediction:
            conf.segmentation_name = ""
            if conf.postprocess_smooth:
                conf.segmentation_name += "smooth_"
            conf.segmentation_name += conf.postprocess_method
            if conf.postprocess_prune is not None:
                conf.segmentation_name += f"_{conf.postprocess_prune}"

    if conf.num_gpus > 1 and conf.gpu is None and conf.local_rank is not None:
        conf.distributed = True

    if conf.distributed:
        conf.device = f"cuda:{conf.local_rank}"
        torch.cuda.set_device(conf.local_rank)
        torch.distributed.init_process_group(
            backend=conf.distribution_backend,
            world_size=conf.num_gpus,
            rank=conf.local_rank,
        )
        conf.world_size = torch.distributed.get_world_size()
        conf.rank = torch.distributed.get_rank()

    if conf.local_rank is None:
        conf.logger = True
    else:
        conf.logger = conf.local_rank == 0

    if conf.logger:
        conf = create_output_directory(conf)

    t_max = conf.max_steps - conf.switch_schedule
    if conf.schedule_1 == "cosine":
        conf.schedule_1_config["t_max"] = t_max
    if conf.schedule_2 == "cosine":
        conf.schedule_2_config["t_max"] = t_max

    setup_logger(conf.verbose, conf.log_file)
    global log
    log = logging.getLogger("config")

    log.info(f"Start from device {conf.device} with local rank {conf.local_rank}")

    # Copy things to log dir
    if conf.logger:
        for path in conf.src_dir.rglob("*.py"):
            out_path = conf.log_src_dir.joinpath(path.relative_to(conf.src_dir))
            if not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                assert out_path.suffix == ".py"
                shutil.copy(path, out_path)
        if not conf.log_dir.joinpath(conf.input_data_path.name).exists():
            shutil.copy(conf.input_data_path, conf.log_dir)

    check_setup(conf)

    # We have ensured in check_setup() that overwrite is either true or false if output
    # probability maps exist. If they don't exist, overwrite should not matter and we
    # set it to false just in case. Change from str to bool for ease.
    if conf.overwrite in ["y", "yes"]:
        conf.overwrite = True
    else:
        conf.overwrite = False

    # Hide other gpus than the one you want to use. For separate inference jobs on
    # multiple-gpu machines.
    if conf.gpu is not None:
        conf.device = f"cuda:{conf.gpu}"
        torch.cuda.set_device(conf.gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(conf.gpu)
    else:
        if not conf.train_mode:
            conf.num_gpus = 0
            conf.device = "cpu"

    if conf.rng_seed is None:
        # Non-deterministic run plus random seed
        conf.rng_seed = torch.initial_seed()
    else:
        # Deterministic run plus manually chosen seed
        # Commented the below instruction since not all operations used have
        # deterministic alternatives
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    np.random.seed(conf.rng_seed % (2**32 - 1) + conf.rank)
    torch.manual_seed(conf.rng_seed + conf.rank)

    # Dump config
    if conf.logger:
        log.info(f"Writing runlog to: {conf.log_file}")

        if not conf.config_file.exists():
            log.info(f"Writing config to: {conf.config_file}")
            with conf.config_file.open("w") as f:
                sorted_config = OrderedDict()
                attrs = {k: v for k, v in vars(conf).items() if not k.startswith("_")}
                for key, val in sorted(attrs.items()):
                    sorted_config[key] = val
                _ = toml.dump(sorted_config, f)

    return conf


def create_output_directory(conf: Configurations) -> Configurations:
    if conf.restore_path is not None:
        # filename is assumed to be on the form state-checkpoint_step-{step:06}.pth
        conf.restore_step = int(conf.restore_path.stem.split("-")[-1])

    if conf.train_mode:
        train_set_name = conf.input_data_path.stem
        if conf.restore_path is None:
            assert conf.output_dir is not None, "--output_dir must be specified"
            root_path = conf.output_dir
        else:
            if conf.max_steps <= conf.restore_step:
                raise ValueError(
                    f"Max steps {conf.max_steps} must be greater than the restored "
                    f"step {conf.restore_step}"
                )
            if conf.output_dir is None:
                resume_folder = "resumed_{:05d}".format(conf.restore_step)
                root_path = conf.restore_path.parents[1].joinpath(resume_folder)
                root_path = root_path.joinpath(conf.config_name)
            else:
                root_path = conf.output_dir
        run_parent_dir = root_path.joinpath(train_set_name)
        run_parent_dir.mkdir(parents=True, exist_ok=True)
        run_dir = utils.make_rundir(run_parent_dir, conf.start_time)

        conf.debug_dir = run_dir.joinpath("debug")
        conf.checkpoint_dir = run_dir.joinpath("checkpoints")
        conf.log_dir = run_dir.joinpath("logs")
        conf.log_src_dir = conf.log_dir.joinpath("src")
        conf.checkpoint_dir.mkdir(exist_ok=True)
        conf.log_src_dir.mkdir(parents=True, exist_ok=True)
    else:
        test_set_name = conf.input_data_path.stem
        if conf.output_dir is None:
            inference_dir = conf.restore_path.parents[1].joinpath("inference")
        else:
            inference_dir = conf.output_dir
        set_dir = inference_dir.joinpath(test_set_name)
        run_parent_dir = set_dir.joinpath("results")
        run_parent_dir.mkdir(parents=True, exist_ok=True)
        run_dir = run_parent_dir.joinpath(conf.config_name)
        run_dir = run_dir.joinpath("checkpoint_{:06d}".format(conf.restore_step))
        if conf.run_dir is not None:
            run_dir = conf.run_dir

        conf.probability_map_dir = run_dir.joinpath("probability_maps_class-255")
        conf.debug_dir = run_dir.joinpath("debug")
        conf.log_dir = run_dir.joinpath("logs")
        conf.log_src_dir = conf.log_dir.joinpath("src")
        conf.log_src_dir.mkdir(parents=True, exist_ok=True)

    conf.log_file = conf.log_dir.joinpath("runlog.txt")
    conf.config_file = conf.log_dir.joinpath("config.toml")

    return conf


def check_setup(conf: Configurations):

    if conf.num_gpus > 0:
        assert conf.batch_size % conf.num_gpus == 0, (
            f"batch_size {conf.batch_size} is not an integer multiplum of "
            f"num_gpus {conf.num_gpus}"
        )

    if conf.distributed:
        assert (
            torch.distributed.is_initialized()
        ), f"Distributed is not initialised on local rank {conf.local_rank}"
        if conf.distribution_backend == "nccl":
            assert (
                torch.distributed.is_nccl_available()
            ), f"NCCL is not available on local rank {conf.local_rank}"

    if not conf.train_mode:
        assert conf.restore_path is not None, "restore is not set"

    loss_functions = ["cross_entropy", "dice", "ce_top10", "ce_top50", "ce_top90"]
    assert conf.loss_function_1 in loss_functions
    if conf.loss_function_2 is not None:
        assert conf.loss_function_2 in loss_functions

    if not conf.train_mode:
        if conf.probability_map_dir.exists():
            empty = len(list(conf.probability_map_dir.rglob("*.png"))) == 0
        else:
            empty = True
        if not empty and conf.overwrite is None:
            log.error(f"Output exists and is non-empty: {conf.probability_map_dir}")
            log.error("Decide what to do with --overwrite")
            sys.exit()

        if conf.merge_tile_predictions and not conf.inference_per_scan:
            log.warning(
                "Configuration 'merge_tile_predictions=True' will not have any effect "
                "since 'inference_per_scan' is False"
            )

        if conf.inference_per_scan and conf.batch_size != 1:
            log.error(f"Batch size is {conf.batch_size} in per-scan inference mode")
            sys.exit()

        if conf.full_image_path is not None and not conf.merge_tile_predictions:
            log.warning(
                "--full_image will not have any effect since "
                "'merge_tile_predictions' is 'False'"
            )


def level_from_verbose(verbose_level: int = 0) -> int:
    if verbose_level <= 0:
        level = logging.CRITICAL
    elif verbose_level == 1:
        level = logging.ERROR
    elif verbose_level == 2:
        level = logging.WARNING
    elif verbose_level == 3:
        level = logging.INFO
    else:
        level = logging.DEBUG
    return level


def setup_logger(verbose_level: int = 0, logfile: Optional[Path] = None):
    fmt = logging.Formatter(
        fmt="[%(asctime)s][%(name)-10.10s][%(levelname)-1.1s] %(message)s",
        datefmt="%Y.%m.%d %H:%M:%S",
    )
    logging.root.setLevel(level_from_verbose(verbose_level))

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(fmt)
    logging.root.addHandler(console_handler)

    if logfile is not None:
        file_handler = logging.FileHandler(filename=str(logfile), mode="a")
        file_handler.setFormatter(fmt)
        logging.root.addHandler(file_handler)
