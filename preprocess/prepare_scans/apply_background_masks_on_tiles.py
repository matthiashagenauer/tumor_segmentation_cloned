import sys
import argparse
from pathlib import Path
from collections import namedtuple

import cv2  # type: ignore
from tqdm import tqdm  # type: ignore
import numpy as np


Case = namedtuple("Case", ["tile", "mask", "output"])
Coord = namedtuple("Coord", ["row", "col"])


def find_corresponding_cases(
    tile_folders, tile_root, mask_paths, mask_root, source_appendix
):
    cases = []
    mask_paths = sorted(mask_paths)
    for tile_folder in tile_folders:
        section = tile_folder.parent.name
        mask_path = mask_root.joinpath(tile_folder.relative_to(tile_root)).with_name(
            f"{section}{source_appendix}.png"
        )
        output_folder = tile_folder.with_name(section + "_mask-foreground-annotation")
        if mask_path in mask_paths:
            cases.append(Case(tile_folder, mask_path, output_folder))
        else:
            print("ERROR: Could not find corresponding paths")
            print(f"Tile folder:   {tile_folder}")
            print(f"Expected mask: {mask_path}")
            sys.exit()
    return cases


def filter_existing(cases, overwrite):
    included_cases = []
    for case in cases:
        include = False
        if case.output.exists():
            if overwrite is None:
                print(f"Path exists: {case.output}")
                print("Decide what to do using --overwrite. Aborting")
                sys.exit()
            else:
                include = overwrite
        else:
            include = True
        if include:
            included_cases.append(case)
    return included_cases


def tile_coords_from_name(name):
    """
    Expected name:

    <section>_mask-annotation_rows-<min row>-<max row>_cols-<min col>-<max col>
    """
    rows_part = name.split("_")[-2]
    cols_part = name.split("_")[-1]
    rows_split = rows_part.split("-")
    cols_split = cols_part.split("-")
    start = Coord(int(rows_split[1]), int(cols_split[1]))
    end = Coord(int(rows_split[2]), int(cols_split[2]))
    return [start, end]


def process_case(case):
    tile_paths = list(case.tile.glob("*_mask-annotation_rows-*_cols-*.png"))
    tile_coords = {p.name: tile_coords_from_name(p.stem) for p in tile_paths}
    height = max([c[1].row for c in tile_coords.values()])
    width = max([c[1].col for c in tile_coords.values()])

    foreground_mask = cv2.imread(str(case.mask), cv2.IMREAD_GRAYSCALE)
    foreground_mask = ((foreground_mask > 0) * 255).astype(np.uint8)
    foreground_mask = cv2.resize(
        foreground_mask, (width, height), interpolation=cv2.INTER_AREA
    )
    foreground_mask = foreground_mask > 127

    case.output.mkdir(parents=True, exist_ok=True)
    for input_name, coords in tile_coords.items():
        start_coord, end_coord = coords
        output_name = input_name.replace(
            "_mask-annotation", "_mask-foreground-annotation"
        )
        foreground_tile = foreground_mask[
            start_coord.row : end_coord.row, start_coord.col : end_coord.col
        ]
        annotation_tile = (
            cv2.imread(str(case.tile.joinpath(input_name)), cv2.IMREAD_GRAYSCALE) > 127
        )
        result_tile = np.zeros(foreground_tile.shape, dtype=np.uint8)
        result_tile[foreground_tile & ~annotation_tile] = 127
        result_tile[foreground_tile & annotation_tile] = 255
        cv2.imwrite(
            str(case.output.joinpath(output_name).with_suffix(".png")), result_tile
        )


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "tiles",
        metavar="PATH",
        help="Root folder where tiles reside",
    )
    parser.add_argument(
        "masks",
        metavar="PATH",
        help="Root folder where background masks reside",
    )
    parser.add_argument(
        "--overwrite",
        metavar="STR",
        choices=["y", "yes", "n", "no"],
        help="If result file exist, overwrite (y/yes) or skip (n/no)",
    )
    args = parser.parse_args()

    source_appendix = "_mask-foreground-annotation"

    tile_root = Path(args.tiles)
    tile_folders = list(tile_root.rglob("*_mask-annotation"))
    print(f"Found {len(tile_folders)} tile folders")
    mask_root = Path(args.masks)
    mask_paths = list(mask_root.rglob(f"*{source_appendix}.png"))
    print(f"Found {len(mask_paths)} foreground masks")
    overwrite = None if args.overwrite is None else args.overwrite in ["y", "yes"]

    cases = find_corresponding_cases(
        tile_folders, tile_root, mask_paths, mask_root, source_appendix
    )
    print(f"Found {len(cases)} corresponding cases")
    cases = filter_existing(cases, overwrite)
    print(f"Running {len(cases)} corresponding cases")

    for case in tqdm(cases):
        process_case(case)


if __name__ == "__main__":
    main()
