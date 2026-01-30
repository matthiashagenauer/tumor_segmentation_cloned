import sys
import argparse
import logging
from enum import Enum

from grid import TilingMode, tiling_mode_from_string

log = logging.getLogger()


class TileFormat(Enum):
    PNG = 1
    JPEG = 2


def valid_tile_formats():
    return [".png", ".jpg"]


def tile_format_from_string(string: str) -> TileFormat:
    if string in ["png", ".png"]:
        tile_format = TileFormat.PNG
    elif string in ["jpg", ".jpg", "jpeg", ".jpeg"]:
        tile_format = TileFormat.JPEG
    else:
        log.error(f"Unimplemented tile format: {string}")
        sys.exit()
    return tile_format


class Configuration:
    def __init__(self, args: argparse.Namespace):
        self.mode = args.mode
        self.verbose = args.verbose

        self.tile_format = tile_format_from_string(args.format)
        self.read_tile_size = 512
        self.target_mpp = args.mpp
        self.target_height = args.size
        self.target_width = args.size
        self.tiling_mode = tiling_mode_from_string(args.mode)
        self.min_overlap = 0 if args.min_overlap is None else args.min_overlap

        self.use_cache = args.cache is not None
        self.tile_workers = args.tile_workers
        self.copy_workers = args.copy_workers

        self.tile_annotation = False
        self.overwrite = None if args.overwrite is None else args.overwrite == "yes"
        self.discard_homogeneous_tiles = [0] if args.discard_zero_valued_tiles else []

        self.output_root = args.output
        self.cache_root = "/tmp" if args.cache is None else args.cache

        # If --mask is given, set mask = union(mask_image == mask_foreground_i), and
        # only include tile if
        # - for area_threshold = 0        # at least 1 pixel in tile must be foreground
        #   sum(mask[tile_region]) > 0
        # - for area_threshold in (0, 1)  # a fraction of pixels must be foreground
        #   sum(mask[tile_region]) / area(tile_region) > area_threshold
        # - for area_threshold = 1        # all pixels in tile must be foreground
        #   sum(mask[tile_region]) == area(tile_region)
        self.tile_inside_mask = False
        self.area_threshold = 0.5
        self.mask_foreground = [255]
        self.mask_mpp = 5

        self.check_config()

    def check_config(self):
        if self.tile_annotation and self.tiling_mode == TilingMode.OUTSIDE:
            # The last row and the last column may be out of bounds. Pad
            # these tiles where appropriate
            log.error("FIXME: TilingMode.OUTSIDE with annotations is unimplented")
            sys.exit()
