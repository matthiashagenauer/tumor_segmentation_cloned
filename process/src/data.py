import logging
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Dict

import pandas as pd  # type: ignore
import cv2  # type: ignore
import numpy as np
from torch.utils.data import Dataset
import albumentations as albu  # type: ignore

import configurations
import utils

log = logging.getLogger("data")


def get_data_paths(
    input_path: Path,
    output_root: Optional[Path] = None,
    overwrite: bool = False,
    merge_tile_predictions: Optional[bool] = False,
) -> pd.DataFrame:
    """
    dataset_filepath is the fileapth of csv file that defines the dataset.

    Example of file used for training:

    ImagePath,MaskPath
    path/to/image_1.png,path/to/corresponding_mask_1.png
    path/to/image_2.png,path/to/corresponding_mask_2.png

    Example of file used for inference:

    ImagePath
    path/to/image_1.png
    path/to/image_2.png

    """
    df = pd.read_csv(str(input_path))
    df["ImagePath"] = df["ImagePath"].apply(Path)
    df["ImageFolder"] = df["ImagePath"].apply(lambda p: p.parent)
    if output_root is not None:
        input_root = utils.common_path(list(df["ImagePath"].values))
        # Move root one folder up if there is only one source scan to make it equal to
        # when there are more than one source scan
        if input_root == df["ImagePath"].iloc[0].parent:
            input_root = input_root.parent
        df["Output"] = df["ImagePath"].apply(
            lambda p: utils.relative_path(p, input_root, output_root)
        )
        df["Output"] = df["Output"].apply(lambda p: p.with_suffix(".png"))
        df["Exists"] = df["Output"].apply(lambda p: p.exists())
        if np.any(df["Exists"].values):
            log.info(f"Output root exists and is non-empty: {output_root}")
            assert overwrite is not None, "--overwrite must be set"
            if overwrite:
                log.info("Overwriting existing results")
            else:
                log.info("Only process input with non-existing output")
                log.info(f"Num input before filtering: {len(df):>6}")
                num_exists = sum(df["Exists"].values)
                num_not_exists = len(df) - num_exists
                if merge_tile_predictions:
                    log.info(f"Use existing results on:    {num_exists:>6}")
                else:
                    df = df[~df["Exists"]]
                    assert num_not_exists == len(df)
                log.info(f"Run network inference on:   {num_not_exists:>6}")
    return df


class SegmentedImages(Dataset):
    def __init__(
        self,
        dataset_df: pd.DataFrame,
        classes: Sequence[int],
        train_mode: bool,
        min_divisor: int,
        transform: Optional[Callable] = None,
        preprocessing_fn: Optional[Callable] = None,
    ):
        self.dataset_df = dataset_df
        self.train_mode = train_mode
        self.min_divisor = min_divisor
        self.classes = classes
        self.mask_label_encoding = True  # label (N, H, W) or one-hot (N, C, H, W)
        self.image_paths = list(self.dataset_df["ImagePath"].values)
        self.has_reference = "MaskPath" in self.dataset_df.columns
        self.transform = transform
        self.preprocessing = self.get_preprocessing(preprocessing_fn)

    def __len__(self) -> int:
        return self.dataset_df.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.dataset_df.iloc[index]
        image_filepath = str(entry.loc["ImagePath"])
        image = cv2.imread(image_filepath, cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        if not self.train_mode:
            image = pad_to_divisible(image, self.min_divisor)

        if self.has_reference:
            mask_filepath = str(entry.loc["MaskPath"])
            mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
            if not self.train_mode:
                mask = pad_to_divisible(mask, self.min_divisor)
            if self.mask_label_encoding:
                # Set all labels not in self.classes to 0
                for value in set(np.unique(mask)).difference(set(self.classes)):
                    mask[mask == value] = 0
                for label, value in enumerate(self.classes):
                    mask[mask == value] = label
                mask = mask.astype(np.int64)
            else:
                mask = np.stack([mask == v for v in self.classes], axis=-1).astype(
                    "float"
                )
            if self.transform is not None:
                temp = self.transform(image=image, mask=mask)
                image = temp["image"]
                mask = temp["mask"]
            if self.preprocessing is not None:
                temp = self.preprocessing(image=image, mask=mask)
                image = temp["image"]
                mask = temp["mask"]
            sample = {
                "image": image,
                "mask": mask,
                "height": height,
                "width": width,
                "image_path": image_filepath,
                "mask_path": mask_filepath,
            }
        else:
            if self.transform is not None:
                temp = self.transform(image=image)
                image = temp["image"]
            if self.preprocessing is not None:
                temp = self.preprocessing(image=image)
                image = temp["image"]
            sample = {
                "image": image,
                "height": height,
                "width": width,
                "image_path": image_filepath,
            }

        return sample

    def get_preprocessing(
        self, preprocessing_fn: Optional[Callable]
    ) -> Optional[albu.Compose]:
        """
        By default, preprocessing_fn is from `src/encoders_init.py` which calls a
        function `preprocess_input` from `smp.encoders._preprocessing` with parameters
        set in `configurations.py`.

        These parameters are `input_space`, `input_range`, `mean` and `std`. If
        `pretrained` is not None, it uses parameters from the pretrained run (e.g.
        imagenet mean and std). If `pretrained` is None, it uses parameters from
        `configurations.py`.

        This function (in this order)
        - transpose the input from bgr to rgb if `input_space == BGR`
        - divides input by 255 if input.max() > 1 and input_range[1] == 1
        - subtracts input by mean
        - divide input by std
        and then returns input
        """
        if preprocessing_fn is None:
            return None
        if self.mask_label_encoding:
            transform = [
                albu.Lambda(image=preprocessing_fn),
                albu.Lambda(image=transpose_to_float32, mask=to_int64),
            ]
        else:
            transform = [
                albu.Lambda(image=preprocessing_fn),
                albu.Lambda(image=transpose_to_float32, mask=transpose_to_int64),
            ]
        return albu.Compose(transform)


def get_training_transform(conf: configurations.Configurations) -> albu.Compose:
    transform = []
    if conf.distort_basic:
        transform.extend(
            [
                albu.HorizontalFlip(p=0.5),
                albu.RandomRotate90(p=0.5),
            ]
        )
    transform.append(
        albu.PadIfNeeded(
            min_height=conf.target_height,
            min_width=conf.target_width,
            always_apply=True,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
        )
    )
    if conf.distort_morphology:
        # FIXME: Set input height somewhere else and check for it?
        # NOTE: RandomResizedCrop works w.r.t. input image, and we first want to crop it
        # down to around target size.
        # NOTE: Assumes square input and target.
        input_height = 2048
        min_scale = 0.9
        max_scale = 1.1
        target_factor = conf.target_height / input_height
        transform.extend(
            [
                albu.RandomResizedCrop(
                    height=conf.target_height,
                    width=conf.target_width,
                    scale=(min_scale * target_factor, max_scale * target_factor),
                    ratio=(4 / 5, 5 / 4),
                    always_apply=True,
                    p=1.0,
                ),
                albu.GaussNoise(
                    var_limit=[0.0, 10.0],
                    mean=0,
                    per_channel=True,
                    always_apply=False,
                    p=0.5,
                ),
                albu.GaussianBlur(
                    blur_limit=[3, 7],
                    sigma_limit=0,
                    always_apply=False,
                    p=0.5,
                ),
            ]
        )
    else:
        if conf.distort_basic:
            transform.append(
                albu.RandomCrop(
                    height=conf.target_height,
                    width=conf.target_width,
                    always_apply=True,
                )
            )
        else:
            transform.append(
                albu.CenterCrop(
                    height=conf.target_height,
                    width=conf.target_width,
                    always_apply=True,
                )
            )
    if conf.distort_bc:
        transform.append(
            albu.RandomBrightnessContrast(
                brightness_limit=[-0.2, 0.2],
                contrast_limit=[-0.2, 0.2],
                brightness_by_max=True,
                always_apply=True,
                p=1.0,
            )
        )
    if conf.distort_hsv:
        transform.append(
            albu.HueSaturationValue(
                hue_shift_limit=[-26, 26],
                sat_shift_limit=[-26, 26],
                val_shift_limit=[-26, 26],
                always_apply=True,
                p=1.0,
            )
        )
    return albu.Compose(transform)


def get_test_transform(conf: configurations.Configurations) -> albu.Compose:
    return albu.Compose([])


def transpose_to_float32(x: np.ndarray, **kwargs) -> np.ndarray:
    return x.transpose(2, 0, 1).astype("float32")


def transpose_to_int64(x: np.ndarray, **kwargs) -> np.ndarray:
    return x.transpose(2, 0, 1).astype("int64")


def to_int64(x: np.ndarray, **kwargs) -> np.ndarray:
    return x.astype("int64")


def make_divisible(current_size: int, min_divisor: int) -> int:
    if current_size % min_divisor == 0:
        new_size = current_size
    else:
        new_size = current_size + (min_divisor - current_size % min_divisor)
    return new_size


def pad_to_divisible(image: np.ndarray, min_divisor: int) -> np.ndarray:
    height = image.shape[0]
    width = image.shape[1]
    new_height = make_divisible(height, min_divisor)
    new_width = make_divisible(width, min_divisor)
    if new_height != height or new_width != width:
        image = albu.augmentations.functional.pad(
            image,
            min_height=new_height,
            min_width=new_width,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
        )
    return image
