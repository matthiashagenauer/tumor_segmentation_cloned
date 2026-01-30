import sys
import numpy as np
import logging

from common import background_filter, common_utils

log = logging.getLogger()


def recompute_mask(out_path, mask_classes, overwrite):
    postfix = common_utils.get_mask_postfix(mask_classes)
    write_path = out_path.with_name(out_path.stem + postfix + ".png")
    return recompute(write_path, overwrite)


def recompute(result_path, overwrite):
    if result_path.exists() and overwrite is None:
        log.error(f"Result path exists, pass --overwrite y/yes or n/no: {result_path}")
        sys.exit()
    return overwrite if result_path.exists() else True


def filter_existing_scans(scan_paths, scan_root, out_root):
    # NOTE: Unused for now, but keeping it for reference
    existing_result_paths, _ = common_utils.find_files([out_root], [".png"])
    existing_relative_paths = []
    log.info("Existing")
    for path in existing_result_paths:
        relative_path = path.relative_to(out_root).with_suffix("")
        existing_relative_paths.append(relative_path)
    existing_paths = []
    not_existing_paths = []
    log.info("Check")
    for scan_path in scan_paths:
        stem = common_utils.format_filestem(scan_path)
        relative_path = scan_path.relative_to(scan_root).parent.joinpath(stem)
        if relative_path in existing_relative_paths:
            existing_paths.append(scan_path)
        else:
            not_existing_paths.append(scan_path)
    log.info(f"Found {len(existing_paths)} existing scans")
    return not_existing_paths


def mask_image_by_value(image, value):
    if len(image.shape) == 2:
        return image == value
    else:
        assert image.shape[-1] == len(value)
        assert image.shape[-1] in [3, 4]
        mask_0 = image[:, :, 0] == value[0]
        mask_1 = image[:, :, 1] == value[1]
        mask_2 = image[:, :, 2] == value[2]
        mask = mask_0 & mask_1 & mask_2
        if image.shape[-1] == 4:
            mask_3 = image[:, :, 3] == value[3]
            mask = mask & mask_3
        return mask


def bbox(foreground_mask):
    row_coords, col_coords = np.where(foreground_mask > 0)
    min_row = np.min(row_coords)
    max_row = np.max(row_coords) + 1
    min_col = np.min(col_coords)
    max_col = np.max(col_coords) + 1
    return min_row, max_row, min_col, max_col


def crop_image(image, mask=None, mpp=None):
    height, width, _ = image.shape
    if mask is None:
        assert mpp is not None, "Please provide mpp"
        mask = background_filter(image, mpp=mpp)
    min_row, max_row, min_col, max_col = bbox(mask)
    image = image[min_row:max_row, min_col:max_col]
    mask = mask[min_row:max_row, min_col:max_col]
    info = {}
    info["original_height"] = int(height)
    info["original_width"] = int(width)
    info["crop_min_row"] = int(min_row)
    info["crop_max_row"] = int(max_row)
    info["crop_min_col"] = int(min_col)
    info["crop_max_col"] = int(max_col)
    return image, mask, info
