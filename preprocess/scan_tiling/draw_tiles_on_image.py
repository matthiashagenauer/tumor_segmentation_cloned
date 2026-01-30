import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import cv2
from tqdm import tqdm

from common import common_utils
from grid import create_grid, tiling_mode_from_string

log = logging.getLogger()


class Configuration:
    def __init__(self, args):
        self.fill_color = (0, 255, 0)  # BGR
        self.line_color = (0, 0, 255)  # BGR
        self.add_contour = args.hide_contour is False

        # If tiles are input, use tile coordinates to draw a grid
        self.base_mpp = args.base_mpp
        self.tile_mpp = args.tile_mpp

        # If no tiles are input, a grid covering the image is drawn
        self.tile_size = args.size
        self.grid_mode = tiling_mode_from_string(args.mode)
        self.min_overlap = args.min_overlap

        # Resize if needed
        if args.resize is not None:
            self.tile_size = int(args.resize * self.tile_size)
            self.min_overlap = int(args.resize * self.min_overlap)
            self.base_mpp = self.base_mpp / args.resize


def coordinates_from_path(path):
    row_start = int(path.stem.split("_")[-2].split("-")[1])
    row_stop = int(path.stem.split("_")[-2].split("-")[2])
    col_start = int(path.stem.split("_")[-1].split("-")[1])
    col_stop = int(path.stem.split("_")[-1].split("-")[2])
    return range(row_start, row_stop), range(col_start, col_stop)


def grid_from_tiles(tile_dir, conf):
    paths = list(tile_dir.glob("*.jpg"))
    tile_contours = []
    resize_factor = conf.tile_mpp / conf.base_mpp
    for path in paths:
        row, col = coordinates_from_path(path)
        row = range(int(row.start * resize_factor), int(row.stop * resize_factor))
        col = range(int(col.start * resize_factor), int(col.stop * resize_factor))
        tile_contours.append((row, col))
    return tile_contours


def draw_tile_contour(image, tile_contours, line_color, line_thickness):
    for tile_contour in tile_contours:
        row = tile_contour[0]
        col = tile_contour[1]
        polygon = np.array(
            [
                [col.start, row.start],  # Top left
                [col.stop, row.start],  # Top right
                [col.stop, row.stop],  # Bottom right
                [col.start, row.stop],  # Bottom left
            ]
        )
        cv2.drawContours(image, [polygon.astype(int)], 0, line_color, line_thickness)
    return image


def draw_tile_image(tile_contours, bgr_color, height, width):
    hsv_color = cv2.cvtColor(
        np.array(bgr_color)[np.newaxis, np.newaxis, :].astype(np.uint8),
        cv2.COLOR_BGR2HSV,
    )[0, 0, :]
    hue_value = hsv_color[0]
    image_hue = np.ones((height, width)) * hue_value
    image_sat = np.zeros((height, width))
    image_val = np.ones((height, width)) * 255
    for tile_contour in tile_contours:
        row = tile_contour[0]
        col = tile_contour[1]
        image_sat[row.start : row.stop, col.start : col.stop] += 50
    if np.min(image_sat) < np.max(image_sat):
        image_sat = (
            (image_sat - np.min(image_sat)) / (np.max(image_sat) - np.min(image_sat))
        ) * 255
    image_hsv = np.dstack((image_hue, image_sat, image_val)).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    return image_bgr, image_sat


def process(
    image_paths,
    image_root,
    tile_root,
    output_root,
    conf,
):
    for image_path in tqdm(image_paths):
        relative_path = image_path.relative_to(image_root)
        output_path = output_root.joinpath(relative_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image_height, image_width, _ = image.shape

        if tile_root is not None:
            tile_dir = tile_root.joinpath(relative_path).with_suffix("")
            tile_contours = grid_from_tiles(tile_dir, conf)
        else:
            row_ranges, col_ranges = create_grid(
                image_height,
                image_width,
                conf.tile_size,
                conf.tile_size,
                conf.min_overlap,
                conf.grid_mode,
            )
            tile_contours = []
            for row in row_ranges:
                for col in col_ranges:
                    tile_contours.append((row, col))

        tile_image, mask = draw_tile_image(
            tile_contours, conf.fill_color, image_height, image_width
        )
        merged_image = cv2.addWeighted(image, 0.5, tile_image, 0.5, 0)
        image[mask > 0] = merged_image[mask > 0]

        if conf.add_contour:
            diagonal = np.sqrt(image_height * image_height + image_width * image_width)
            line_thickness = max(1, int(np.round(diagonal / 2000)))
            image = draw_tile_contour(
                image, tile_contours, conf.line_color, line_thickness
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        # cv2.imwrite(
        #     str(output_path.with_name(output_path.stem + "_tiles.png")), tile_image
        # )


def get_paths(args):
    output_root = Path(args.output)
    image_paths, image_root = common_utils.find_files(
        [Path(p) for p in args.input], [".png", ".jpg"]
    )
    tile_root = None if args.tiles is None else Path(args.tiles)

    if image_root == output_root:
        log.error(f"Input root and output root are equal: {image_root}")
        log.error("This will overwrite original images. Terminating")
        sys.exit()

    log.info(f"Found {len(image_paths)} input images")
    return image_paths, image_root, tile_root, output_root


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "input",
        nargs="+",
        metavar="PATH",
        help="Image paths or folders with images. Used as base images",
    )
    parser.add_argument(
        "--tiles",
        metavar="PATH",
        help=(
            "Root path of tiles corresponding to input base images. If this is not "
            "a grid of square tiles are drawn on the entire base image."
        ),
    )
    parser.add_argument("--output", metavar="PATH", required=True, help="Output folder")
    parser.add_argument(
        "--size",
        metavar="INT",
        type=int,
        default=2048,
        help="Sidelength in square result tiles. [default: %(default)d]",
    )
    parser.add_argument(
        "--mode",
        metavar="STR",
        choices=["overlap", "inside", "outside"],
        default="inside",
        help=(
            "Determine how to tile grid if input tiles are not given: \n"
            "- 'overlap': The entire scan is tiled with overlap if required for \n"
            "    fit. --min_overlap can be specified together with this option.\n"
            "- 'inside': The scan is tiled from the top left corner. The last tile \n"
            "    is the last tile that fully fit inside the scan in either direction.\n"
            "- 'outside': The scan is tiled from the top left corner. The last tile \n"
            "    is the first non-filled tile in either direction, except when all \n"
            "    tiles fit perfectly in the scan\n"
            "[default: %(default)s]"
        ),
    )
    parser.add_argument(
        "--min_overlap",
        metavar="INT",
        type=int,
        help=(
            "Require at least this many overlapping pixels in both directions. \n"
            "Only used with --mode == 'overlap'."
        ),
    )
    parser.add_argument(
        "--base_mpp",
        metavar="FLOAT",
        type=float,
        default=5.0,
        help="Base resolution used if --tiles is given. [default: %(default)d]",
    )
    parser.add_argument(
        "--tile_mpp",
        metavar="FLOAT",
        type=float,
        default=1.0,
        help="Tile resolution used if --tiles is given. [default: %(default)d]",
    )
    parser.add_argument(
        "--resize",
        metavar="FLOAT",
        type=float,
        help="Scale the input tile size by this factor",
    )
    parser.add_argument(
        "--hide-contour",
        action="store_true",
        help="Remove tile contour in addition to saturated overlap",
    )
    args = parser.parse_args()

    image_paths, image_root, tile_root, output_root = get_paths(args)
    conf = Configuration(args)

    process(
        image_paths,
        image_root,
        tile_root,
        output_root,
        conf,
    )


if __name__ == "__main__":
    main()
