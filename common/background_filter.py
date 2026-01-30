from pathlib import Path
from typing import Optional, Sequence

import cv2  # type: ignore
import numpy as np
from skimage import morphology, filters  # type: ignore

import common_utils


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    min_val = np.min(image)
    max_val = np.max(image)
    scaled = (image - min_val) / (max_val - min_val)
    return np.floor(scaled * 255).astype(np.uint8)


def new_factor(original_size: int, min_size: int, original_factor: float):
    if original_size * original_factor < min_size:
        new = min_size / original_size
    else:
        new = original_factor
    return new


def scale_factor(
    height: int, width: int, min_height: int, min_width: int, target_factor: float
) -> float:
    height_factor = new_factor(height, min_height, target_factor)
    width_factor = new_factor(width, min_width, target_factor)
    return min(1.0, max(height_factor, width_factor))


def compute_entropy(image, size, mask=None, binary=False):
    # NOTE: Max entropy is 8 for 8-bit grayscale and 2 for 2-bit grayscale
    footprint = np.ones((size, size))
    if mask is not None and mask.shape != image.shape:
        mask = (
            cv2.resize(
                (mask * 255).astype(np.uint8),
                None,
                fx=image.shape[1] / mask.shape[1],
                fy=image.shape[0] / mask.shape[0],
                interpolation=cv2.INTER_AREA,
            )
            > 127
        )

    if binary:
        # assumes values 0 and 255
        binary_mask = image > 127
        n = size * size
        fg_sum = filters.rank.sum(binary_mask.astype(np.uint8), footprint, mask=mask)
        fg_mask = fg_sum > 0
        bg_mask = fg_sum < n
        bg_term = np.zeros(binary_mask.shape, np.float32)
        bg_term[bg_mask] = (1.0 - fg_sum[bg_mask] / n) * np.log2(
            1.0 - fg_sum[bg_mask] / n
        )
        fg_term = np.zeros(binary_mask.shape, np.float32)
        fg_term[fg_mask] = (fg_sum[fg_mask] / n) * np.log2(fg_sum[fg_mask] / n)
        entropy_image = -(bg_term + fg_term) / 2.0
    else:
        entropy_image = filters.rank.entropy(image, footprint, mask=mask) / 8.0
    return entropy_image


def background_filter(
    image_bgr: Optional[np.ndarray] = None,
    input_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    # NOTE: Parameter values are set with the assumption that the input image is a WSI
    # with tissue section(s) and some white-ish background. The image is assumed to be
    # in resolution 5 mpp. This algorithm spend ~1s per image with these assumptions.

    if input_mask is None:
        assert image_bgr is not None, "Image or mask must be input to background_filter"
        mask = (cv2.Canny(image_bgr, 10, 50) > 127).astype(np.uint8)
    else:
        mask = input_mask

    # Remove small regions
    min_bg_area = 10000
    min_fg_area = 1600
    s_size = 9
    s_size = s_size + (1 if s_size % 2 == 0 else 0)
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (s_size, s_size))

    # Remove small background regions
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, structuring_element)
    mask = (
        morphology.remove_small_holes(mask > 0, min_bg_area, connectivity=1)
    ).astype(np.uint8)

    # Remove small foreground regions
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, structuring_element)
    mask = (
        morphology.remove_small_objects(mask > 0, min_fg_area, connectivity=1)
    ).astype(np.uint8)

    return mask


def filter_annotation_mask(
    annotation_mask: np.ndarray,
    image_bgr: np.ndarray,
    out_path: Path,
    mask_classes: Sequence[str],
):
    height, width, _ = image_bgr.shape
    assert height == annotation_mask.shape[0], "Unequal mask and image height"
    assert width == annotation_mask.shape[1], "Unequal mask and image width"
    foreground_mask = background_filter(image_bgr)

    annotation_mask = annotation_mask > 0
    foreground_mask = foreground_mask > 0

    out_mask = create_mask(mask_classes, annotation_mask, foreground_mask, image_bgr)
    write_path = common_utils.get_mask_path(out_path, mask_classes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(write_path), out_mask)


def create_mask(
    mask_classes: Sequence[str],
    annotation_mask: Optional[np.ndarray] = None,
    input_foreground_mask: Optional[np.ndarray] = None,
    image_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    if annotation_mask is not None:
        height, width = annotation_mask.shape
    elif input_foreground_mask is not None:
        height, width = input_foreground_mask.shape
    elif image_bgr is not None:
        height, width, _ = image_bgr.shape
    else:
        raise ValueError(
            "All of annotation_mask, foreground_mask and image_bgr are None"
        )

    if input_foreground_mask is None:
        assert image_bgr is not None, "image_bgr must be provided"
        foreground_mask = background_filter(image_bgr)
        foreground_mask = foreground_mask > 0
    else:
        foreground_mask = input_foreground_mask

    if "annotation" in mask_classes:
        assert annotation_mask is not None, "annotation_mask must be provided"

    out_mask = np.zeros((height, width), dtype=np.uint8)
    if "annotation" in mask_classes and "foreground" not in mask_classes:
        out_mask[annotation_mask] = 255
        out_mask[~foreground_mask] = 0
    if "annotation" not in mask_classes and "foreground" in mask_classes:
        out_mask[foreground_mask] = 127
        out_mask[~foreground_mask] = 0
    if "annotation" in mask_classes and "foreground" in mask_classes:
        out_mask[foreground_mask] = 127
        out_mask[annotation_mask] = 255
        out_mask[~foreground_mask] = 0
    return out_mask
