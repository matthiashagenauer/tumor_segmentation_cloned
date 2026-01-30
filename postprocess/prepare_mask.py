import sys

import cv2
import numpy as np
from skimage import filters


def structuring_element(size):
    size = size + (1 if size % 2 == 0 else 0)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def connected_components(mask):
    _, label_image = cv2.connectedComponents(mask.astype(np.uint8))
    fg_labels = np.unique(label_image[mask > 0])
    return label_image, fg_labels


def sharpening(image):
    image = np.clip(image, 0, 255).astype(np.uint8)
    blur_image = cv2.GaussianBlur(image, (101, 101), 0)
    unsharp = cv2.addWeighted(image, 2.5, blur_image, -1.5, 0)
    return unsharp


def resized_dim(height, width, target_factor, min_area):
    """
    Resize height and widht with the same factor. This factor is set to target_factor
    unless the resulting area is smaller than min_area, in which case the factor will be
    greater.
    """
    # Copy from utils
    area = height * width
    assert area > 0
    factor = max(target_factor, np.sqrt(min_area / area))
    target_height = int(np.floor(factor * height))
    target_width = int(np.floor(factor * width))
    return target_height, target_width


def prepare_probability_map(image):
    target_factor = 0.2
    min_area = 1e6
    height, width = image.shape
    target_height, target_width = resized_dim(height, width, target_factor, min_area)
    image = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

    image = cv2.medianBlur(image, 9)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    image = np.clip(image, 0, 255).astype(np.uint8)

    image = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )

    return image


def prune_regions(mask, probability_map, method):
    assert method.startswith("percentile"), f"Invalid prune method format: {method}"
    if len(method.split("-")) == 1:
        percentile = 50
        threshold = 229
    elif len(method.split("-")) == 3:
        percentile = int(method.split("-")[-2])
        threshold = int(method.split("-")[-1])
    else:
        print(f"ERROR: Invalid prune method: {method}")
        sys.exit()
    assert 0 < percentile < 100, f"Invalid prune percentile value: {percentile}"
    assert 0 <= threshold < 255, f"Invalid prune threshold value: {percentile}"
    label_image, fg_labels = connected_components(mask)
    new_mask = np.zeros_like(mask)
    for label in fg_labels:
        region_mask = label_image == label
        values = probability_map[region_mask]
        if np.percentile(values, percentile) > threshold:
            new_mask[region_mask] = True
    return new_mask


def prepare_mask(
    probability_map,
    image_bgr,
    smooth,
    method,
    prune,
    gpu,
):
    height, width = probability_map.shape

    orig_probability_map = probability_map
    if smooth:
        probability_map = prepare_probability_map(probability_map)

    if method == "argmax":
        mask = probability_map > 127
    elif method == "crf":
        # Import here since it use torch which is not allways availalable, and
        # this feature is seldomly used.
        # Very annoying to get an import error from a part of the program that is not in
        # use.
        import crf
        mask = crf.segment_probability(image_bgr, probability_map, gpu)
    elif method.startswith("threshold"):
        if len(method.split("-")) == 2:
            value = int(method.split("-")[-1])
            size = None
        elif len(method.split("-")) == 3:
            value = int(method.split("-")[-2])
            size = int(method.split("-")[-1])
        else:
            print(f"ERROR: Expected threshold-<value>[-<value], got {method}")
        assert 0 <= value < 255, f"Expected value in [0, 255), got: {value}"
        new_mask = probability_map > value
        if size is not None:
            selem = structuring_element(size)
            mask = cv2.morphologyEx(new_mask.astype(np.uint8), cv2.MORPH_DILATE, selem)
            mask = mask > 0
        else:
            mask = new_mask
    elif method.startswith("dilate"):
        threshold_inner = int(method.split("-")[-3])
        threshold_outer = int(method.split("-")[-2])
        size = int(method.split("-")[-1])
        assert (
            threshold_inner >= threshold_outer
        ), "Outer threshold must be smaller than inner"
        assert size % 2 == 1, "Dilation size must be odd"
        inner_mask = probability_map > threshold_inner
        outer_mask = probability_map > threshold_outer
        selem = structuring_element(size)
        dilated_mask = (
            cv2.morphologyEx(inner_mask.astype(np.uint8), cv2.MORPH_DILATE, selem) > 0
        )
        mask = dilated_mask & outer_mask
    elif method.startswith("dist"):
        threshold_inner = int(method.split("-")[-3])  # (0, 255)
        threshold_dist = int(method.split("-")[-2])  # Pixels. Ex 5 mpp: 200 px = 1mm
        threshold_product = int(method.split("-")[-1])  # (0, 255)
        assert 0 < threshold_inner < 255, f"Inner threshold: {threshold_inner}"
        assert 0 <= threshold_dist, f"Distance treshold: {threshold_dist}"
        assert 0 < threshold_product < 255, f"Product treshold: {threshold_product}"
        inner_mask = probability_map <= threshold_inner
        dist_image = cv2.distanceTransform(
            (inner_mask).astype(np.uint8),
            distanceType=cv2.DIST_L2,
            maskSize=cv2.DIST_MASK_3,
        )
        dist_image = 1.0 - np.clip(dist_image / threshold_dist, 0, 1)
        product_image = dist_image * (probability_map.astype(float) / 255.0)
        product_image = np.clip(np.floor(product_image * 255), 0, 255).astype(np.uint8)
        mask = product_image > threshold_product
    elif method.startswith("hysteresis"):
        threshold_lower = int(method.split("-")[-2])  # (0, 255)
        threshold_higher = int(method.split("-")[-1])  # (0, 255)
        assert 0 < threshold_lower < 255, f"Lower threshold: {threshold_lower}"
        assert 0 < threshold_higher < 255, f"Higher treshold: {threshold_higher}"
        hysteresis_mask = filters.apply_hysteresis_threshold(
            probability_map, threshold_lower, threshold_higher
        )
        mask = hysteresis_mask.astype(bool)
    elif method.startswith("reconstruction"):
        if len(method.split("-")) == 1:
            mask_threshold = 178
            seed_threshold = 229
        elif len(method.split("-")) == 3:
            mask_threshold = int(method.split("-")[-2])
            seed_threshold = int(method.split("-")[-1])
        else:
            print(f"ERROR: Invalid mask generating method: {method}")
        rec_mask = (probability_map > mask_threshold).astype(np.uint8)
        rec_seed = probability_map > seed_threshold
        label_image, fg_labels = connected_components(rec_mask > 0)
        new_mask = np.zeros_like(rec_seed)
        for label in fg_labels:
            region_mask = label_image == label
            if (region_mask & rec_seed).any():
                new_mask[region_mask] = True
        selem = structuring_element(15)
        mask = cv2.morphologyEx(new_mask.astype(np.uint8), cv2.MORPH_DILATE, selem)
        mask = mask > 0
    else:
        print(f"ERROR: Invalid mask generating method: {method}")
        sys.exit()

    if prune is not None:
        mask = prune_regions(mask, orig_probability_map, prune)

    return mask, probability_map
