import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

import cv2  # type: ignore
import numpy as np

log = logging.getLogger("tile")

TRange2D = TypeVar("TRange2D", bound="Range2D")
TTile = TypeVar("TTile", bound="Tile")


class Range2D:
    def __init__(
        self,
        name: Optional[str] = None,
        vertical: Optional[range] = None,
        horisontal: Optional[range] = None,
    ):
        if vertical is not None and horisontal is not None:
            self.vertical = vertical
            self.horisontal = horisontal
        elif name is not None:
            self.from_file_name(name)
        else:
            log.error("'vertical' and 'horisontal' or 'name' is required for Range2D")
            sys.exit()

        self.top = self.vertical.start
        self.bottom = self.vertical.stop
        self.left = self.horisontal.start
        self.right = self.horisontal.stop

        assert self.bottom > self.top
        assert self.right > self.left

        self.height = self.bottom - self.top
        self.width = self.right - self.left
        self.dim = (self.height, self.width)

    def __repr__(self) -> str:
        return (
            f"Row [{self.top:>5}, {self.bottom:>5}) "
            f"Col [{self.left:>5}, {self.right:>5})"
        )

    def __eq__(self, other) -> bool:
        top = hasattr(other, "top") and self.top == other.top
        bottom = hasattr(other, "bottom") and self.bottom == other.bottom
        left = hasattr(other, "left") and self.left == other.left
        right = hasattr(other, "right") and self.right == other.right
        return top and bottom and left and right

    def __hash__(self) -> int:
        return hash(str(self))

    def filename(self) -> str:
        return (
            f"row-{self.top:05d}-{self.bottom:05d}_"
            f"col-{self.left:05d}-{self.right:05d}"
        )

    def from_file_name(self, name: str):
        """
        Expects

        name = <something>_rows-<row start>-<row end>_cols-<col start>-<col end>
        """
        parts = name.split("_")
        row_part = [p for p in parts if p.startswith("rows")][0]
        col_part = [p for p in parts if p.startswith("cols")][0]
        self.vertical = range(
            int(row_part.split("-")[-2]), int(row_part.split("-")[-1])
        )
        self.horisontal = range(
            int(col_part.split("-")[-2]), int(col_part.split("-")[-1])
        )

    def resize(
        self,
        factor: float,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        new_top = int(self.top * factor)
        if height is not None:
            assert height > 0
            new_bottom = new_top + height
        else:
            new_bottom = int(self.bottom * factor)
        new_left = int(self.left * factor)
        if width is not None:
            assert width > 0
            new_right = new_left + width
        else:
            new_right = int(self.right * factor)
        return Range2D(
            vertical=range(new_top, new_bottom), horisontal=range(new_left, new_right)
        )

    def is_overlapping(self, other: TRange2D) -> bool:
        def range_is_overlapping(r1: range, r2: range) -> bool:
            return r2.start in r1 or r1.start in r2

        return range_is_overlapping(
            self.vertical, other.vertical
        ) and range_is_overlapping(self.horisontal, other.horisontal)

    def overlap_top(self, other: TRange2D) -> Optional[range]:
        if self.is_overlapping(other):
            if (
                self.top > other.top
                and self.top < other.bottom
                and self.bottom > other.bottom
            ):
                return range(self.top, other.bottom)
            else:
                return None
        else:
            return None

    def overlap_bottom(self, other: TRange2D) -> Optional[range]:
        if self.is_overlapping(other):
            if (
                other.top > self.top
                and other.top < self.bottom
                and other.bottom > self.bottom
            ):
                return range(other.top, self.bottom)
            else:
                return None
        else:
            return None

    def overlap_left(self, other: TRange2D) -> Optional[range]:
        if self.is_overlapping(other):
            if (
                self.left > other.left
                and self.left < other.right
                and self.right > other.right
            ):
                return range(self.left, other.right)
            else:
                return None
        else:
            return None

    def overlap_right(self, other: TRange2D) -> Optional[range]:
        if self.is_overlapping(other):
            if (
                other.left > self.left
                and other.left < self.right
                and other.right > self.right
            ):
                return range(other.left, self.right)
            else:
                return None
        else:
            return None

    def direct_overlap_top(self, other: TRange2D) -> Optional[range]:
        if self.horisontal != other.horisontal:
            return None
        if other.top < self.top and other.bottom in self.vertical:
            return range(self.top, other.bottom)
        else:
            return None

    def direct_overlap_bottom(self, other: TRange2D) -> Optional[range]:
        if self.horisontal != other.horisontal:
            return None
        if other.bottom > self.bottom and other.top in self.vertical:
            return range(other.top, self.bottom)
        else:
            return None

    def direct_overlap_left(self, other: TRange2D) -> Optional[range]:
        if self.vertical != other.vertical:
            return None
        if other.left < self.left and other.right in self.horisontal:
            return range(self.left, other.right)
        else:
            return None

    def direct_overlap_right(self, other: TRange2D) -> Optional[range]:
        if self.vertical != other.vertical:
            return None
        if other.right > self.right and other.left in self.horisontal:
            return range(other.left, self.right)
        else:
            return None

    def is_direct_overlap_top(self, other: TRange2D) -> bool:
        return self.direct_overlap_top(other) is not None

    def is_direct_overlap_bottom(self, other: TRange2D) -> bool:
        return self.direct_overlap_bottom(other) is not None

    def is_direct_overlap_left(self, other: TRange2D) -> bool:
        return self.direct_overlap_left(other) is not None

    def is_direct_overlap_right(self, other: TRange2D) -> bool:
        return self.direct_overlap_right(other) is not None


class OverlappingRange2D:
    def __init__(self, this_range: Range2D, other_ranges: Sequence[Range2D]):
        self.range2d = this_range
        self.overlapping_ranges_top: List[range] = []
        self.overlapping_ranges_bottom: List[range] = []
        self.overlapping_ranges_left: List[range] = []
        self.overlapping_ranges_right: List[range] = []
        for other_range in other_ranges:
            if this_range == other_range or not this_range.is_overlapping(other_range):
                continue
            else:
                overlap_top_range = this_range.overlap_top(other_range)
                if overlap_top_range is not None:
                    self.overlapping_ranges_top.append(overlap_top_range)
                overlap_bottom_range = this_range.overlap_bottom(other_range)
                if overlap_bottom_range is not None:
                    self.overlapping_ranges_bottom.append(overlap_bottom_range)
                overlap_left_range = this_range.overlap_left(other_range)
                if overlap_left_range is not None:
                    self.overlapping_ranges_left.append(overlap_left_range)
                overlap_right_range = this_range.overlap_right(other_range)
                if overlap_right_range is not None:
                    self.overlapping_ranges_right.append(overlap_right_range)

    def max_overlap_top(self) -> Optional[range]:
        if len(self.overlapping_ranges_top) > 0:
            return max(self.overlapping_ranges_top, key=lambda x: len(x))
        else:
            return None

    def max_overlap_bottom(self) -> Optional[range]:
        if len(self.overlapping_ranges_bottom) > 0:
            return max(self.overlapping_ranges_bottom, key=lambda x: len(x))
        else:
            return None

    def max_overlap_left(self) -> Optional[range]:
        if len(self.overlapping_ranges_left) > 0:
            return max(self.overlapping_ranges_left, key=lambda x: len(x))
        else:
            return None

    def max_overlap_right(self) -> Optional[range]:
        if len(self.overlapping_ranges_right) > 0:
            return max(self.overlapping_ranges_right, key=lambda x: len(x))
        else:
            return None

    def max_overlap_top_size(self) -> Optional[int]:
        m = self.max_overlap_top()
        return None if m is None else len(m)

    def max_overlap_bottom_size(self) -> Optional[int]:
        m = self.max_overlap_bottom()
        return None if m is None else len(m)

    def max_overlap_left_size(self) -> Optional[int]:
        m = self.max_overlap_left()
        return None if m is None else len(m)

    def max_overlap_right_size(self) -> Optional[int]:
        m = self.max_overlap_right()
        return None if m is None else len(m)


class Tile:
    def __init__(
        self,
        range2d: Range2D,
        image: np.ndarray,
        overlap_info: Optional[OverlappingRange2D] = None,
    ):
        self.range2d = range2d
        self.image = image
        self.overlap_info = overlap_info
        if overlap_info is not None:
            assert self.range2d == overlap_info.range2d

    def direct_overlap_top(self, other: TTile) -> Optional[range]:
        return self.range2d.direct_overlap_top(other.range2d)

    def direct_overlap_bottom(self, other: TTile) -> Optional[range]:
        return self.range2d.direct_overlap_bottom(other.range2d)

    def direct_overlap_left(self, other: TTile) -> Optional[range]:
        return self.range2d.direct_overlap_left(other.range2d)

    def direct_overlap_right(self, other: TTile) -> Optional[range]:
        return self.range2d.direct_overlap_right(other.range2d)

    def is_overlapping(self, other: TTile) -> bool:
        return self.range2d.is_overlapping(other.range2d)

    def max_overlap_top(self) -> Optional[range]:
        if self.overlap_info is not None:
            return self.overlap_info.max_overlap_top()
        else:
            return None

    def max_overlap_bottom(self) -> Optional[range]:
        if self.overlap_info is not None:
            return self.overlap_info.max_overlap_bottom()
        else:
            return None

    def max_overlap_left(self) -> Optional[range]:
        if self.overlap_info is not None:
            return self.overlap_info.max_overlap_left()
        else:
            return None

    def max_overlap_right(self) -> Optional[range]:
        if self.overlap_info is not None:
            return self.overlap_info.max_overlap_right()
        else:
            return None

    def max_overlap_top_size(self) -> Optional[int]:
        if self.overlap_info is not None:
            return self.overlap_info.max_overlap_top_size()
        else:
            return None

    def max_overlap_bottom_size(self) -> Optional[int]:
        if self.overlap_info is not None:
            return self.overlap_info.max_overlap_bottom_size()
        else:
            return None

    def max_overlap_left_size(self) -> Optional[int]:
        if self.overlap_info is not None:
            return self.overlap_info.max_overlap_left_size()
        else:
            return None

    def max_overlap_right_size(self) -> Optional[int]:
        if self.overlap_info is not None:
            return self.overlap_info.max_overlap_right_size()
        else:
            return None


def construct_tiles(
    input_paths: Sequence[Path],
    tile_predictions: Sequence[np.ndarray],
    resize_factor: Optional[float] = None,
) -> Dict[Path, Tile]:
    assert len(input_paths) > 0
    ranges = [Range2D(name=p.stem) for p in input_paths]
    if resize_factor is not None:
        log.debug(f"Resize tiles with factor {resize_factor:.4f}")
        # Ensure all ranges have the same dimension
        templ = ranges[0].resize(resize_factor)
        ranges = [r.resize(resize_factor, templ.height, templ.width) for r in ranges]
    overlap_info_dict = {r: OverlappingRange2D(r, ranges) for r in ranges}
    tiles: Dict[Path, Tile] = {}
    for path, image, range2d in zip(input_paths, tile_predictions, ranges):
        if resize_factor is not None and resize_factor != 1.0:
            image = cv2.resize(
                image,
                (range2d.width, range2d.height),
                interpolation=cv2.INTER_AREA,
            )
        tiles[path] = Tile(range2d, image, overlap_info_dict[range2d])
    return tiles


def global_dim_from_ranges(ranges: Sequence[Range2D]) -> Tuple[int, int]:
    assert len(ranges) > 0
    height = max([r.bottom for r in ranges])
    width = max([r.right for r in ranges])
    return (height, width)
