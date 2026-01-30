import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from skimage import morphology
from tqdm import tqdm


def mm_squared_to_pixels(area, resolution):
    area_per_pixel = resolution * resolution
    return int(np.floor(area / area_per_pixel))


# def pixels_to_mm_squared(pixels, resolution):
#     area_per_pixel = resolution * resolution
#     return pixels * area_per_pixel


# def pixels_to_diameter(pixels, resolution):
#     area = pixels_to_mm_squared(pixels, resolution)
#     return 2 * np.sqrt(area / np.pi)


# def pixels_to_square_length(pixels, resolution):
#     area = pixels_to_mm_squared(pixels, resolution)
#     return np.sqrt(area)


def diameter_to_pixels(diameter, resolution):
    area = np.pi * ((diameter / 2.0) ** 2)
    return mm_squared_to_pixels(area, resolution)


def process(input_paths, input_root, output_root, min_fg_diameter, resolution, value):
    min_fg_pixels = diameter_to_pixels(min_fg_diameter, resolution)
    print(f"Min foreground diameter (mm): {min_fg_diameter:>9.2f}")
    print(f"Min foreground pixel count:   {min_fg_pixels:>6}")
    for input_path in tqdm(input_paths):
        output_path = output_root.joinpath(input_path.relative_to(input_root))
        if output_path.exists():
            print(f"ERROR: Skipping since output path exists: {output_path}")
            continue
        image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        mask = image == 255
        remaining_mask = morphology.remove_small_objects(mask, min_fg_pixels)
        image[np.logical_xor(mask, remaining_mask)] = value
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)


def check_cli(args):
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input path does not exist {input_path}")
        sys.exit()
    if input_path.is_file() and input_path.suffix != ".png":
        print(f"ERROR: Input file must be png {input_path}")
        sys.exit()
    output_path = Path(args.output)
    if output_path.suffix != "":
        print(f"ERROR: Output path cannot be file {output_path}")
        sys.exit()
    if not isinstance(args.value, int):
        print(f"ERROR: Input --value must be integer, is: {args.value.dtype}")
        sys.exit()
    if args.value < 0 or args.value > 255:
        print(f"ERROR: Input --value must be in range [0, 255], is: {args.value}")
        sys.exit()
    return input_path, output_path


def remove_existing(input_paths, input_root, output_root):
    keep_input_paths = []
    for input_path in input_paths:
        output_path = output_root.joinpath(input_path.relative_to(input_root))
        if output_path.exists():
            continue
        keep_input_paths.append(input_path)
    return keep_input_paths


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "input",
        metavar="PATH",
        help="Input root folder or image",
    )
    parser.add_argument("output", metavar="PATH", help="Output root folder")
    parser.add_argument(
        "--diameter",
        metavar="FLOAT",
        type=float,
        default=0.5,
        help=(
            "Minimum foreground area given as a circle with diameter --diameter. "
            "Specified in millimetres. Default = 0.5"
        ),
    )
    parser.add_argument(
        "--resolution",
        metavar="INT",
        type=int,
        default=5,
        help=(
            "Resolution of input masks given in micro metre per square pixel "
            "sidelength. Default=5"
        ),
    )
    parser.add_argument(
        "--value",
        type=int,
        default=0,
        help="Value used to replace the removed regions in [0, 255]",
    )
    args = parser.parse_args()

    resolution = args.resolution / 1000.0  # milli metre per pixel sidelength
    min_fg_diameter = args.diameter

    input_path, output_root = check_cli(args)
    value = args.value

    if input_path.is_file():
        input_root = input_path.parent
        input_paths = [input_path]
    else:
        input_root = input_path
        input_paths = list(input_root.rglob("*.png"))

    print(f"Input {len(input_paths)} images")
    input_paths = remove_existing(input_paths, input_root, output_root)

    print(f"Processing {len(input_paths)} images after filtering existing")
    process(input_paths, input_root, output_root, min_fg_diameter, resolution, value)


if __name__ == "__main__":
    main()
