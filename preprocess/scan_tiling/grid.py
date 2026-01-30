from enum import Enum
import logging
import sys
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger()


class TilingMode(Enum):
    # The scan is tiled from the top left corner with overlap (if required) to ensure
    # that the entire scan is covered. The overlap is approximately equal between all
    # adjacent tiles. Minimum allowed overlap can be specified.
    OVERLAP = 1
    # The scan is tiled from the top left corner without overlap. The last tile is the
    # last tile that fully fit inside the scan in either direction.
    INSIDE = 2
    # The scan is tiled from the top left corner without overlap. The last tile is the
    # first non-filled tile in either direction, except when all tiles fit perfectly in
    # the scan.
    OUTSIDE = 3
    # The scan is tiled from the top left corner without overlap. If the tile grid does
    # not match perfectly the scan in one direction, the last set of tiles in this
    # direction will be smaller than the target size in this direction. With this we
    # have non-overlapping tiles where the leftmost and bottommost tiles are potentially
    # smaller in width or height, respectively so that the tiles together cover the scan
    # perfectly.
    REST = 4


def tiling_mode_from_string(string: str) -> TilingMode:
    if string == "overlap":
        tiling_mode = TilingMode.OVERLAP
    elif string == "inside":
        tiling_mode = TilingMode.INSIDE
    elif string == "outside":
        tiling_mode = TilingMode.OUTSIDE
    elif string == "rest":
        tiling_mode = TilingMode.REST
    else:
        log.error(f"Unimplemented tiling mode: {string}")
        sys.exit()
    return tiling_mode


def find_overlap(full_size: int, part_size: int, min_overlap: int) -> Tuple[float, int]:
    if full_size <= part_size:
        log.error(
            f"Part size '{part_size}' must be smaller than " f"full size '{full_size}'"
        )
        sys.exit()
    num_parts = int(np.ceil(full_size / part_size))
    overlap = (part_size * num_parts - full_size) / (num_parts - 1)
    if overlap < min_overlap:
        if part_size <= min_overlap:
            log.error(
                f"Part size '{part_size}' must be greater than "
                f"minimal overlap '{min_overlap}'. "
                f"Computed number of parts '{num_parts}' and overlap '{overlap}' "
                f"with full size '{full_size}'"
            )
            sys.exit()
        num_parts = int(np.ceil((full_size - min_overlap) / (part_size - min_overlap)))
        overlap = (part_size * num_parts - full_size) / (num_parts - 1)
    return overlap, num_parts


def divide_with_overlap(
    full_size: int, part_size: int, min_overlap: int
) -> List[range]:
    """
    Divide a full line into parts where the line have size full_size and the parts have
    size part_size (except when part_size > full_size). Return a list of part start
    (inclusive) and end (exclusive) points on the full line.

    If part_size is larger than full_size, return a part with size full_size. If not
    return a list of parts where each part overlaps the other with a minimum overlap of
    min_overlap. The first parts will overlap with the same amount (above or equal to
    min_overlap), but the penultimate and last part may overlap more so that the end
    point of the last part is equal to the end point on the full line.
    """
    ranges: List[range] = []
    if full_size > part_size:
        overlap, num_parts = find_overlap(full_size, part_size, min_overlap)
        fractional_part = overlap - np.floor(overlap)
        num_ceils = int(np.floor(num_parts * fractional_part))
        for k in range(num_parts):
            if k <= num_ceils:
                int_overlap = int(np.ceil(overlap))
            else:
                int_overlap = int(np.floor(overlap))
            if k == 0:
                start = 0
            else:
                start = ranges[k - 1].stop - int_overlap
            stop = start + part_size
            ranges.append(range(start, stop))
    else:
        ranges.append(range(0, full_size))
    return ranges


def divide_without_overlap(full_size: int, part_size: int) -> List[range]:
    """
    Divide a full line into parts where the line have size full_size and the parts have
    size part_size. Return a list of part start (inclusive) and end (exclusive) points
    on the full line.

    The end point of the final part is greater than or equal to the end point of the
    full line
    """
    start_points = np.arange(0, full_size, part_size)
    return [range(int(p), int(p + part_size)) for p in start_points]


def overlapping_grid(
    image_height: int,
    image_width: int,
    tile_height: int,
    tile_width: int,
    min_overlap: int,
) -> Tuple[List[range], List[range]]:
    """
    Return coords ([(start_row, end_row), ...], [(start_col, end_col), ...]) of tiles
    """
    row_ranges = divide_with_overlap(image_height, tile_height, min_overlap)
    col_ranges = divide_with_overlap(image_width, tile_width, min_overlap)
    return row_ranges, col_ranges


def nonoverlapping_grid(
    image_height: int, image_width: int, tile_height: int, tile_width: int
) -> Tuple[List[range], List[range]]:
    """
    Return coords ([(start_row, end_row), ...], [(start_col, end_col), ...]) of tiles
    """
    row_ranges = divide_without_overlap(image_height, tile_height)
    col_ranges = divide_without_overlap(image_width, tile_width)
    return row_ranges, col_ranges


def create_grid(
    image_height: int,
    image_width: int,
    tile_height: int,
    tile_width: int,
    min_overlap: Optional[int],
    tiling_mode: TilingMode,
) -> Tuple[List[range], List[range]]:
    """
    Create x (row) and y (col) coordinates for tile corners.
    """
    if tiling_mode == TilingMode.OVERLAP:
        assert min_overlap is not None
        row_ranges, col_ranges = overlapping_grid(
            image_height,
            image_width,
            tile_height,
            tile_width,
            min_overlap,
        )
    else:
        row_ranges, col_ranges = nonoverlapping_grid(
            image_height,
            image_width,
            tile_height,
            tile_width,
        )
        if tiling_mode == TilingMode.INSIDE:
            if row_ranges[-1].stop > image_height:
                row_ranges = row_ranges[:-1]
            if col_ranges[-1].stop > image_width:
                col_ranges = col_ranges[:-1]
        elif tiling_mode == TilingMode.REST:
            if row_ranges[-1].stop > image_height:
                row_ranges[-1] = range(row_ranges[-1].start, image_height)
            if col_ranges[-1].stop > image_width:
                col_ranges[-1] = range(col_ranges[-1].start, image_width)
    return row_ranges, col_ranges
