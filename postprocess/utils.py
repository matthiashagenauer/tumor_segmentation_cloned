import sys
import logging
from collections import namedtuple

import numpy as np


Paths = namedtuple(
    "Paths", ["input", "image", "reference", "resize", "mask", "probability"]
)


def check_existing(path, overwrite):
    if path.exists():
        if overwrite is None:
            logging.info(f"Output exists: {path}")
            logging.info("Pass --overwrite y(es) or --overwrite n(o)")
            sys.exit()
        elif overwrite in ["y", "yes"]:
            return True
        else:
            return False
    else:
        return True


def find_corresponding(
    input_path,
    input_root,
    image_root,
    mask_root,
    prob_root,
    overwrite,
):
    rel = input_path.relative_to(input_root)
    if image_root is None:
        image = None
        resize = None
        reference = None
    else:
        if image_root.suffix == ".png":
            image = image_root
            resize = None
        else:
            assert image_root.is_dir(), f"Image root is not a directory: {image_root}"
            image = image_root.joinpath(rel)
            resize = image_root.joinpath(rel).with_suffix(".json")
        if image.with_name(image.stem + "_mask-foreground.png").exists():
            reference = image.with_name(image.stem + "_mask-foreground.png")
        elif image.with_name(image.stem + "_mask-foreground-annotation.png").exists():
            reference = image.with_name(image.stem + "_mask-foreground-annotation.png")
        elif image.with_name(image.stem + "_mask-annotation-foreground.png").exists():
            reference = image.with_name(image.stem + "_mask-annotation-foreground.png")
        else:
            reference = None

    if mask_root is not None:
        if mask_root.suffix == ".png":
            mask = mask_root
        else:
            mask = mask_root.joinpath(rel)
        proceed = check_existing(mask, overwrite)
    else:
        mask = None
    if prob_root is not None:
        if prob_root.suffix == ".png":
            prob = prob_root
        else:
            prob = prob_root.joinpath(rel)
        proceed = check_existing(prob, overwrite)
    else:
        prob = None

    if proceed:
        return Paths(input_path, image, reference, resize, mask, prob)
    else:
        return None


def class_index_to_label(mask, class_labels):
    labels = np.unique(mask)
    assert np.max(labels) < len(class_labels), "Incompatible label and classes"
    for label in sorted(labels):
        mask[mask == label] = class_labels[label]
    return mask


def resized_dim(height, width, target_factor, min_area):
    """
    Resize height and widht with the same factor. This factor is set to target_factor
    unless the resulting area is smaller than min_area, in which case the factor will be
    greater.
    """
    area = height * width
    assert area > 0
    factor = max(target_factor, np.sqrt(min_area / area))
    target_height = int(np.floor(factor * height))
    target_width = int(np.floor(factor * width))
    return target_height, target_width


def sigmoid(x, p):
    """
    For x in (0, 1), create a sigmoid

    f = ax^4 + bx^3 + cx^2 + dx + e

    where

    f(0) = 0
    f(p) = p
    f(1) = 1
    f'(0) = 0
    f'(1) = 1
    """
    a = (2 * p * p - 3 * p + 1) / (p * p * p - 2 * p * p + p)
    b = -2 - 2 * a
    c = 1 - (a + b)
    return a * x * x * x * x + b * x * x * x + c * x * x
