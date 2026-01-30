import sys
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np

from configuration import Configuration, TileFormat
from corresponding import Case

from common import scan_utils
from grid import create_grid, TilingMode

try:
    import openslide  # type: ignore
except OSError as err:
    print("Failed to load openslide:")
    print(err)
    print("Try to load it with e.g:")
    print("'$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib'")
    sys.exit()

log = logging.getLogger()


def get_image_tile(
    scan: openslide.OpenSlide,
    read_level: int,
    read_factor: float,
    row_range: range,
    col_range: range,
    target_tile_height: int,
    target_tile_width: int,
) -> np.ndarray:
    # Default background value is used in empty regions only
    # - if the scan file contain empty regions
    # - and if openslide property PROPERTY_NAME_BACKGROUND_COLOR is not present
    # Alpha channel 0 is used to identify empty regions
    # 204 = 0.8 * 255. (204, 204, 204) = #CCCCCC
    bg_value = "CCCCCC"
    # Read region from scan
    read_tile_height = int(np.round(len(row_range) / read_factor))
    read_tile_width = int(np.round(len(col_range) / read_factor))
    read_tile = scan_utils.read_region(
        scan,
        row_range.start,
        col_range.start,
        read_level,
        read_tile_height,
        read_tile_width,
        bg_value,
    )

    # Do extra resizing to fit output shape exactly
    target_tile = cv2.resize(
        read_tile,
        (target_tile_width, target_tile_height),
        interpolation=cv2.INTER_AREA,
    )

    return target_tile


def create_output_path(
    target_row_range: range,
    target_col_range: range,
    output_dir: Path,
    tile_format: TileFormat,
) -> Path:
    suffix = "rows-{:06}-{:06}_cols-{:06}-{:06}".format(
        target_row_range.start,
        target_row_range.stop,
        target_col_range.start,
        target_col_range.stop,
    )
    if tile_format == TileFormat.PNG:
        suffix += ".png"
    elif tile_format == TileFormat.JPEG:
        suffix += ".jpg"
    else:
        log.error(f"ERROR: Unreachable: Invalid tile format: {tile_format}")
        sys.exit()
    return output_dir.joinpath(f"{output_dir.name}_{suffix}")


def write_tile(output_path: Path, target_tile: np.ndarray):
    if output_path.suffix == ".png":
        write_param = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    elif output_path.suffix == ".jpg":
        write_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
    else:
        log.error(f"ERROR: Unreachable: Invalid tile format: {output_path.suffix}")
        sys.exit()
    cv2.imwrite(str(output_path), target_tile, write_param)


def scale_range(source: range, factor: float, limit: int):
    dest = range(int(source.start * factor), int(source.stop * factor))
    if dest.stop > limit:
        raise ValueError(f"Scaled range stop > limit: '{dest.stop} > {limit}'")
    return dest


def shift_range(source: range, addend: int, limit: int):
    dest = range(source.start + addend, source.stop + addend)
    if dest.stop > limit:
        raise ValueError(f"Shifted range stop > limit: '{dest.stop} > {limit}'")
    return dest


def ensure_fit(
    target_row_range: range,
    target_col_range: range,
    target_scan_height: int,
    target_scan_width: int,
) -> Tuple[range, range]:
    row_end = min(target_row_range.stop, target_scan_height)
    row_start = max(0, row_end - len(target_row_range))
    col_end = min(target_col_range.stop, target_scan_width)
    col_start = max(0, col_end - len(target_col_range))
    return range(row_start, row_end), range(col_start, col_end)


def ensure_fit_multi(
    row_ranges: Sequence[range],
    col_ranges: Sequence[range],
    height: int,
    width: int,
) -> Tuple[List[range], List[range]]:
    temp_row_ranges = []
    temp_col_ranges = []
    for row in row_ranges:
        for col in col_ranges:
            row, col = ensure_fit(row, col, height, width)
            temp_row_ranges.append(row)
            temp_col_ranges.append(col)
    return temp_row_ranges, temp_col_ranges


def is_tile_inside_mask(
    mask: np.ndarray,
    target_row_range: range,
    target_col_range: range,
    target_mask_factor: float,
    area_threshold: float,
):
    mask_height = mask.shape[0]
    mask_width = mask.shape[1]
    mask_row_range = scale_range(target_row_range, target_mask_factor, mask_height)
    mask_col_range = scale_range(target_col_range, target_mask_factor, mask_width)
    assert mask_row_range.start >= 0
    assert mask_col_range.start >= 0
    assert mask_row_range.start < mask_row_range.stop
    assert mask_col_range.start < mask_col_range.stop
    mask_region = mask[
        mask_row_range.start : mask_row_range.stop,
        mask_col_range.start : mask_col_range.stop,
    ]
    mask_region_size = mask_region.shape[0] * mask_region.shape[1]
    if area_threshold < 1:
        inside = np.sum(mask_region) / mask_region_size > area_threshold
    else:
        inside = np.sum(mask_region) == mask_region_size
    return inside


def get_target_ranges(
    output_dir: Path,
    target_scan_height: int,
    target_scan_width: int,
    target_tile_height: int,
    target_tile_width: int,
    conf: Configuration,
    mask: Optional[np.ndarray] = None,
    target_mask_factor: Optional[float] = None,
) -> Dict[Path, Tuple[range, range]]:
    # Find coordinates in target domain and upscale them to lvl0 (tiling domain) in
    # order to avoid issues with rounding in edge cases
    target_row_ranges, target_col_ranges = create_grid(
        target_scan_height,
        target_scan_width,
        target_tile_height,
        target_tile_width,
        conf.min_overlap,
        conf.tiling_mode,
    )
    # Fit ranges
    if conf.tiling_mode == TilingMode.OVERLAP:
        target_row_ranges, target_col_ranges = ensure_fit_multi(
            target_row_ranges,
            target_col_ranges,
            target_scan_height,
            target_scan_width,
        )
    # Create list of tile ranges to be tiled
    target_ranges = {}
    for target_row_range in target_row_ranges:
        for target_col_range in target_col_ranges:
            output_path = create_output_path(
                target_row_range,
                target_col_range,
                output_dir,
                conf.tile_format,
            )
            if conf.tile_inside_mask:
                assert mask is not None
                assert target_mask_factor is not None
                if not is_tile_inside_mask(
                    mask,
                    target_row_range,
                    target_col_range,
                    target_mask_factor,
                    conf.area_threshold,
                ):
                    continue
            if output_path.exists() and not conf.overwrite:
                continue
            target_ranges[output_path] = (target_row_range, target_col_range)

    # Write or synchronise file with expected tiles so that one can easily check if some
    # tiles are missing if something goes wrong during tiling
    expected = set([p.name for p in target_ranges.keys()])
    expected_path = output_dir.joinpath("expected_tiles.txt")
    if expected_path.exists():
        with expected_path.open() as f:
            existing = set([r.strip() for r in f.readlines()])
    else:
        existing = set()
    missing = expected.difference(existing)
    if len(missing) > 0:
        with expected_path.open("a") as f:
            for n in sorted(list(missing)):
                f.write(f"{n}\n")
    return target_ranges


def tile_scan_indirectly(
    sample: Case, conf: Configuration, increase_level: bool = False
) -> int:
    """
    Tile the scan with non-overlapping tiles with a fixed size independent of the target
    size. Downscale the tiles to the target resolution and form a complete scan at
    target resolution. Tile this downscaled scan using the input configuration (tile
    size, min overlap, overlap method, etc.).

    This currently cannot be used when tiling inside an annotated region. Use
    tile_scan_directly for this use case. See tile_scan_indirectly for recommendations.
    """
    output_scan_dir = sample.scan_output
    output_scan_dir.mkdir(parents=True, exist_ok=True)

    with openslide.open_slide(str(sample.scan)) as scan:
        try:
            lvl0_mpp_x, lvl0_mpp_y = scan_utils.find_mpp(scan)
        except Exception as e:
            log.error(f"Failed to compute mpp in {sample.scan}")
            log.error(f"{e}")
            return 0

        target_lvl0_factor_x = conf.target_mpp / lvl0_mpp_x
        target_lvl0_factor_y = conf.target_mpp / lvl0_mpp_y

        lvl0_scan_width, lvl0_scan_height = scan.dimensions

        lvl0_nonempty_rect = scan_utils.bounding_rectangle(scan.properties)
        if lvl0_nonempty_rect is not None:
            lvl0_scan_height = lvl0_nonempty_rect.height
            lvl0_scan_width = lvl0_nonempty_rect.width
            lvl0_start_row = lvl0_nonempty_rect.row
            lvl0_start_col = lvl0_nonempty_rect.col
        else:
            lvl0_start_row = 0
            lvl0_start_col = 0

        target_scan_height = int(np.floor(lvl0_scan_height / target_lvl0_factor_y))
        target_scan_width = int(np.floor(lvl0_scan_width / target_lvl0_factor_x))
        target_tile_height = conf.read_tile_size
        target_tile_width = conf.read_tile_size

        # Find coordinates in target domain and upscale them to lvl0 (tiling domain) in
        # order to avoid issues with rounding in edge cases
        target_row_ranges, target_col_ranges = create_grid(
            target_scan_height,
            target_scan_width,
            target_tile_height,
            target_tile_width,
            None,
            TilingMode.REST,
        )
        lvl0_row_ranges = [
            scale_range(r, target_lvl0_factor_y, lvl0_scan_height)
            for r in target_row_ranges
        ]
        lvl0_col_ranges = [
            scale_range(r, target_lvl0_factor_x, lvl0_scan_width)
            for r in target_col_ranges
        ]
        lvl0_row_ranges = [
            shift_range(r, lvl0_start_row, lvl0_scan_height + lvl0_start_row)
            for r in lvl0_row_ranges
        ]
        lvl0_col_ranges = [
            shift_range(r, lvl0_start_col, lvl0_scan_width + lvl0_start_col)
            for r in lvl0_col_ranges
        ]
        num_vertical_tiles = len(lvl0_row_ranges)
        num_horisontal_tiles = len(lvl0_col_ranges)

        target_lvl0_factor = min(target_lvl0_factor_x, target_lvl0_factor_y)
        read_level, _ = scan_utils.find_level(
            target_lvl0_factor, scan.level_downsamples
        )
        if increase_level:
            read_level += 1
            log.warning(f"Increasing read level to {read_level}")
        if read_level >= len(scan.level_downsamples):
            raise ValueError(
                f"Too large read level. Trying to read at level '{read_level}' in scan "
                f"with level downsamples '{scan.level_downsamples}'"
            )
        read_factor = scan.level_downsamples[read_level]

        target_image = np.zeros(
            (target_scan_height, target_scan_width, 3), dtype=np.uint8
        )

        # Tile scan
        count = 0
        for i, lvl0_row_range in enumerate(lvl0_row_ranges):
            target_row_start = i * target_tile_height
            if i == num_vertical_tiles - 1:
                this_tile_height = target_scan_height - target_row_start
            else:
                this_tile_height = target_tile_height
            target_row_end = target_row_start + this_tile_height

            for j, lvl0_col_range in enumerate(lvl0_col_ranges):
                target_col_start = j * target_tile_width
                if j == num_horisontal_tiles - 1:
                    this_tile_width = target_scan_width - target_col_start
                else:
                    this_tile_width = target_tile_width
                target_col_end = target_col_start + this_tile_width

                target_image[
                    target_row_start:target_row_end,
                    target_col_start:target_col_end,
                    :,
                ] = get_image_tile(
                    scan,
                    read_level,
                    read_factor,
                    lvl0_row_range,
                    lvl0_col_range,
                    this_tile_height,
                    this_tile_width,
                )

        # Tile target image
        target_tile_height = min(target_scan_height, conf.target_height)
        target_tile_width = min(target_scan_width, conf.target_width)

        target_ranges = get_target_ranges(
            output_scan_dir,
            target_scan_height,
            target_scan_width,
            target_tile_height,
            target_tile_width,
            conf,
        )
        count = 0
        for output_path, target_range in target_ranges.items():
            row = target_range[0]
            col = target_range[1]
            if row.stop > target_scan_height and conf.tiling_mode != TilingMode.OUTSIDE:
                raise ValueError(
                    "Vertical end point of tile is greater than scan height.\n"
                    "Tiling outside scan is only expected in TilingMode.OUTSIDE"
                )
            if col.stop > target_scan_width and conf.tiling_mode != TilingMode.OUTSIDE:
                raise ValueError(
                    "Horisontal end point of tile is greater than scan width.\n"
                    "Tiling outside scan is only expected in TilingMode.OUTSIDE"
                )
            output_path = create_output_path(
                row, col, output_scan_dir, conf.tile_format
            )
            if output_path.exists() and not conf.overwrite:
                continue
            tile = target_image[row.start : row.stop, col.start : col.stop, :]
            if len(conf.discard_homogeneous_tiles) > 0:
                if not any([np.all(tile == v) for v in conf.discard_homogeneous_tiles]):
                    write_tile(output_path, tile)
                    count += 1
            else:
                write_tile(output_path, tile)
                count += 1

    return count


def tile_scan_directly(
    sample: Case,
    conf: Configuration,
    increase_level: bool = False,
) -> int:
    """
    Specify tiles to tile and read them from the scan.

    Contrast this with tile_scan_indirectly(), which is faster when the whole scan is
    tiled and overlapping tiles are allowed. These methods should be equally fast when
    tiling the whole scan without overlap. This method should be used when target_mpp <
    1 since tile_scan_indirectly() will be prohibitively more memory demanding in that
    case.
    """
    output_scan_dir = sample.scan_output
    output_scan_dir.mkdir(parents=True, exist_ok=True)

    with openslide.open_slide(str(sample.scan)) as scan:
        try:
            lvl0_mpp_x, lvl0_mpp_y = scan_utils.find_mpp(scan)
        except Exception as e:
            log.error(f"Failed to compute mpp in {sample.scan}")
            log.error(f"{e}")
            return 0

        target_lvl0_factor_x = conf.target_mpp / lvl0_mpp_x
        target_lvl0_factor_y = conf.target_mpp / lvl0_mpp_y

        lvl0_scan_width, lvl0_scan_height = scan.dimensions

        lvl0_nonempty_rect = scan_utils.bounding_rectangle(scan.properties)
        if lvl0_nonempty_rect is not None:
            lvl0_scan_height = lvl0_nonempty_rect.height
            lvl0_scan_width = lvl0_nonempty_rect.width
            lvl0_start_row = lvl0_nonempty_rect.row
            lvl0_start_col = lvl0_nonempty_rect.col
        else:
            lvl0_start_row = 0
            lvl0_start_col = 0

        if conf.tile_inside_mask:
            if sample.mask is None:
                raise FileNotFoundError("No mask found but --masks is given")
            mask_lvl0_factor_x = conf.mask_mpp / lvl0_mpp_x
            mask_lvl0_factor_y = conf.mask_mpp / lvl0_mpp_y
            target_mask_factor = conf.target_mpp / conf.mask_mpp
            if sample.mask.suffix == ".png":
                mask_image = cv2.imread(str(sample.mask), cv2.IMREAD_GRAYSCALE)
                # Crude check to see that we don't read "regular" grayscale or colour
                # image, but a mask
                assert len(np.unique(mask_image) < 5)
            else:
                raise ValueError(f"Invalid mask suffix {sample.mask.suffix}")
            mask = np.zeros_like(mask_image, dtype=bool)
            for v in conf.mask_foreground:
                mask += mask_image == v
        else:
            target_mask_factor = None
            mask = None

        target_scan_height = int(np.floor(lvl0_scan_height / target_lvl0_factor_y))
        target_scan_width = int(np.floor(lvl0_scan_width / target_lvl0_factor_x))
        target_tile_height = min(target_scan_height, conf.target_height)
        target_tile_width = min(target_scan_width, conf.target_width)

        target_ranges = get_target_ranges(
            output_scan_dir,
            target_scan_height,
            target_scan_width,
            target_tile_height,
            target_tile_width,
            conf,
            mask,
            target_mask_factor,
        )

        # Tile scan
        target_lvl0_factor = min(target_lvl0_factor_x, target_lvl0_factor_y)
        read_level, _ = scan_utils.find_level(
            target_lvl0_factor, scan.level_downsamples
        )
        if increase_level:
            read_level += 1
            log.warning(f"Increasing read level to {read_level}")
        if read_level >= len(scan.level_downsamples):
            raise ValueError(
                f"Too large read level. Trying to read at level '{read_level}' in scan "
                f"with level downsamples '{scan.level_downsamples}'"
            )
        read_factor = scan.level_downsamples[read_level]
        count = 0
        for output_path, target_range in target_ranges.items():
            target_row_range = target_range[0]
            target_col_range = target_range[1]
            lvl0_row_range = scale_range(
                target_row_range, target_lvl0_factor_y, lvl0_scan_height
            )
            lvl0_col_range = scale_range(
                target_col_range, target_lvl0_factor_x, lvl0_scan_width
            )
            lvl0_row_range = shift_range(
                lvl0_row_range, lvl0_start_row, lvl0_scan_height + lvl0_start_row
            )
            lvl0_col_range = shift_range(
                lvl0_col_range, lvl0_start_col, lvl0_scan_width + lvl0_start_col
            )
            target_scan_tile = get_image_tile(
                scan,
                read_level,
                read_factor,
                lvl0_row_range,
                lvl0_col_range,
                target_tile_height,
                target_tile_width,
            )
            if len(conf.discard_homogeneous_tiles) > 0:
                if not any(
                    [
                        np.all(target_scan_tile == v)
                        for v in conf.discard_homogeneous_tiles
                    ]
                ):
                    write_tile(output_path, target_scan_tile)
                    count += 1
            else:
                write_tile(output_path, target_scan_tile)
                count += 1

    return count


def tile_scan(
    sample: Case,
    conf: Configuration,
    increase_level: bool = False,
) -> int:
    if (
        conf.target_mpp >= 1
        and conf.tiling_mode == TilingMode.OVERLAP
        and not conf.tile_inside_mask
    ):
        tile_count = tile_scan_indirectly(sample, conf, increase_level)
    else:
        tile_count = tile_scan_directly(sample, conf, increase_level)
    return tile_count
