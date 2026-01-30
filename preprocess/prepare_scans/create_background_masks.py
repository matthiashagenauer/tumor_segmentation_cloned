"""
Simple script to create background masks from downscaled input scans.

Background will have value 0 and foreground (preferably tissue) will have value 127.
Filenames will be the original image filename with an appended _mask-foreground.
"""
import sys
import argparse
from pathlib import Path
from collections import namedtuple

import cv2  # type: ignore
from tqdm import tqdm  # type: ignore

from common import common_utils, background_filter


Case = namedtuple("Case", ["input", "output"])


def find_corresponding_cases(input_paths, input_root, output_root, exclude_str):
    cases = []
    for input_path in input_paths:
        if exclude_str is not None and exclude_str in input_path.name:
            continue
        output_path = output_root.joinpath(
            input_path.relative_to(input_root)
        ).with_name(input_path.stem + "_mask-foreground.png")
        cases.append(Case(input_path, output_path))
    return cases


def filter_existing(cases, overwrite):
    included_cases = []
    for case in cases:
        include = False
        if case.output.exists():
            if overwrite is None:
                print("Path exists: {case.output}")
                print("Decide what to do using --overwrite. Aborting")
                sys.exit()
            else:
                include = overwrite
        else:
            include = True
        if include:
            included_cases.append(case)
    return included_cases


def process_case(case):
    image_bgr = cv2.imread(str(case.input), cv2.IMREAD_COLOR)
    mask = background_filter.background_filter(image_bgr) * 127
    case.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(case.output), mask)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "input",
        metavar="PATH",
        nargs="+",
        help="Input image(s) or image root folder(s)",
    )
    parser.add_argument(
        "output",
        metavar="PATH",
        help="Output root folder",
    )
    parser.add_argument(
        "--overwrite",
        metavar="STR",
        choices=["y", "yes", "n", "no"],
        help="If result file exist, overwrite (y/yes) or skip (n/no)",
    )
    parser.add_argument(
        "--exclude",
        metavar="STR",
        help="Exclude files with this word in its filename",
    )
    args = parser.parse_args()

    input_files = [Path(p) for p in args.input]
    image_paths, image_root = common_utils.find_files(input_files, [".png"])
    output_root = Path(args.output)
    overwrite = None if args.overwrite is None else args.overwrite in ["y", "yes"]

    cases = find_corresponding_cases(image_paths, image_root, output_root, args.exclude)
    cases = filter_existing(cases, overwrite)

    for case in tqdm(cases):
        process_case(case)


if __name__ == "__main__":
    main()
