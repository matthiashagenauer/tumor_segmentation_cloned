import logging
import sys
from collections import namedtuple
from typing import Any, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np

try:
    import openslide  # type: ignore
except OSError as err:
    print("Failed to load openslide:")
    print(err)
    print("Try to load it with e.g:")
    print("'$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib'")
    sys.exit()

log = logging.getLogger()


Rectangle = namedtuple("Rectangle", ["row", "col", "height", "width"])


def find_level(
    desired_factor: float, available_factors: Sequence[float]
) -> Tuple[int, float]:
    """
    Return corresponding level and downsampling factor. The downsampleing factor is
    relative to level 0. The returned factor is the largest available factor smaller
    than or equal to the desired factor.

    If the scan is read at this level, one do not need to upsample after to reach the
    desired factor, but one might need to downsample the read region.
    """
    read_level = 0
    read_factor = None
    for level, available_factor in enumerate(available_factors):
        if available_factor > desired_factor:
            break
        read_level = level
        read_factor = available_factor
    return read_level, read_factor


def find_mpp(scan: openslide.OpenSlide) -> Tuple[float, float]:
    if openslide.PROPERTY_NAME_MPP_X in scan.properties.keys():
        lvl0_mpp_x = float(scan.properties[openslide.PROPERTY_NAME_MPP_X])
    elif (
        "tiff.XResolution" in scan.properties.keys()
        and "tiff.ResolutionUnit" in scan.properties.keys()
    ):
        # Converted olympus .tiff scan
        assert scan.properties["tiff.ResolutionUnit"] == "centimeter"
        lvl0_mpp_x = 10000.0 / float(scan.properties["tiff.XResolution"])
    else:
        raise ValueError("No PROPERTY_NAME_MPP_X in scan properties")
    if openslide.PROPERTY_NAME_MPP_Y in scan.properties.keys():
        lvl0_mpp_y = float(scan.properties[openslide.PROPERTY_NAME_MPP_Y])
    elif (
        "tiff.YResolution" in scan.properties.keys()
        and "tiff.ResolutionUnit" in scan.properties.keys()
    ):
        # Converted olympus .tiff scan
        assert scan.properties["tiff.ResolutionUnit"] == "centimeter"
        lvl0_mpp_y = 10000.0 / float(scan.properties["tiff.YResolution"])
    else:
        raise ValueError("No PROPERTY_NAME_MPP_Y in scan properties")
    return lvl0_mpp_x, lvl0_mpp_y


def bounding_rectangle(properties: Mapping[Any, Any]) -> Optional[Rectangle]:
    start_row = None
    start_col = None
    height = None
    width = None
    if openslide.PROPERTY_NAME_BOUNDS_Y in properties.keys():
        start_row = int(properties[openslide.PROPERTY_NAME_BOUNDS_Y])
    if openslide.PROPERTY_NAME_BOUNDS_X in properties.keys():
        start_col = int(properties[openslide.PROPERTY_NAME_BOUNDS_X])
    if openslide.PROPERTY_NAME_BOUNDS_HEIGHT in properties.keys():
        height = int(properties[openslide.PROPERTY_NAME_BOUNDS_HEIGHT])
    if openslide.PROPERTY_NAME_BOUNDS_WIDTH in properties.keys():
        width = int(properties[openslide.PROPERTY_NAME_BOUNDS_WIDTH])
    if start_row is None or start_col is None or height is None or width is None:
        return None
    else:
        return Rectangle(start_row, start_col, height, width)


def include_alpha(image, alpha, bg_value, premultiplied=False):
    """
    According to

    https://openslide.org/docs/premultiplied-argb/

    one should use the premultiplied case. In the few scans I have tested, this does not
    look right in the empty region edges. The non-premultiplied version does, however
    look ok and is similar (to what I can understand) how e.g. DeepZoom handles it, see

    https://github.com/openslide/openslide-python/blob/main/openslide/deepzoom.py

    For premultiplication, see

    https://en.wikipedia.org/wiki/Alpha_compositing
    """
    assert not premultiplied  # See docstring above
    assert image.shape == alpha.shape
    assert isinstance(bg_value, int)
    image = image.copy()
    image[alpha == 0] = bg_value
    # Only change value where not
    # - alpha == 0: set to bg value above
    # - alpha == 255: keep values
    mask = (alpha != 0) & (alpha != 255)
    if premultiplied:
        image[mask] = (255.0 * image[mask] / alpha[mask]).astype(np.uint8)
    else:
        alpha = alpha / 255
        image[mask] = (bg_value * (1 - alpha[mask]) + image[mask] * alpha[mask]).astype(
            np.uint8
        )
    return image


def read_region(
    scan,
    lvl0_start_row,
    lvl0_start_col,
    read_level,
    read_height,
    read_width,
    bg_value_hex_str,
):
    image = np.asarray(
        scan.read_region(
            (lvl0_start_col, lvl0_start_row),  # in level 0
            read_level,
            (read_width, read_height),  # in read level
        )
    )
    bg_value_hex_str = scan.properties.get(
        openslide.PROPERTY_NAME_BACKGROUND_COLOR, bg_value_hex_str
    )
    bg_value = tuple(int(bg_value_hex_str[i : i + 2], 16) for i in [0, 2, 4])
    image_r = include_alpha(image[:, :, 0], image[:, :, 3], bg_value[0])
    image_g = include_alpha(image[:, :, 1], image[:, :, 3], bg_value[1])
    image_b = include_alpha(image[:, :, 2], image[:, :, 3], bg_value[2])
    image = cv2.merge([image_b, image_g, image_r])
    return image


def image_from_scan(scan_path, target_mpp, bg_value_hex_str):
    with openslide.open_slide(str(scan_path)) as scan:
        # Read scan on the level most suitable to the desired output size
        try:
            lvl0_mpp_x, lvl0_mpp_y = find_mpp(scan)
        except Exception as e:
            log.error(f"Failed to compute mpp in {scan_path}")
            log.error(f"{e}")
            return 0

        target_factor_x = target_mpp / lvl0_mpp_x
        target_factor_y = target_mpp / lvl0_mpp_y

        lvl0_width, lvl0_height = scan.dimensions

        lvl0_nonempty_rect = bounding_rectangle(scan.properties)
        if lvl0_nonempty_rect is not None:
            lvl0_height = lvl0_nonempty_rect.height
            lvl0_width = lvl0_nonempty_rect.width
            lvl0_start_row = lvl0_nonempty_rect.row
            lvl0_start_col = lvl0_nonempty_rect.col
        else:
            lvl0_start_row = 0
            lvl0_start_col = 0

        target_height = int(lvl0_height / target_factor_y)
        target_width = int(lvl0_width / target_factor_x)

        target_factor = min(target_factor_x, target_factor_y)
        read_level, read_factor = find_level(target_factor, scan.level_downsamples)
        read_height = int(lvl0_height / read_factor)
        read_width = int(lvl0_width / read_factor)
        if read_width > 20000 or read_height > 20000:
            log.warning(
                f"Reading region at level {read_level} with shape "
                f"{read_height} x {read_width}"
            )
        read_image = read_region(
            scan,
            lvl0_start_row,
            lvl0_start_col,
            read_level,
            read_height,
            read_width,
            bg_value_hex_str,
        )

        # Do extra resizing to fit output shape exactly
        resized_image = cv2.resize(
            read_image, (target_width, target_height), interpolation=cv2.INTER_AREA
        )

        if openslide.PROPERTY_NAME_OBJECTIVE_POWER in scan.properties.keys():
            lvl0_magnification = float(
                scan.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
            )
            target_magnification = lvl0_magnification / target_factor
        else:
            # log.warning(f"No lvl0 magnification in {scan_path}")
            lvl0_magnification = None
            target_magnification = None
        info = {}
        info["lvl0_magnification"] = lvl0_magnification
        info["lvl0_mpp-x"] = lvl0_mpp_x
        info["lvl0_mpp-y"] = lvl0_mpp_y
        info["lvl0_start_row"] = lvl0_start_row
        info["lvl0_start_col"] = lvl0_start_col
        info["lvl0_height"] = lvl0_height
        info["lvl0_width"] = lvl0_width
        info["read_factor"] = read_factor
        info["read_level"] = read_level
        info["read_height"] = read_height
        info["read_width"] = read_width
        info["target_factor"] = target_factor
        info["target_magnification"] = target_magnification
        info["target_mpp"] = target_mpp
        info["target_height"] = target_height
        info["target_width"] = target_width

    return resized_image, info
