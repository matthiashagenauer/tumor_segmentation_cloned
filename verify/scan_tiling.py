"""
A rough check if all expected tiles are in fact tiled.

1. Check that scan tiles are present
2. Check that WSIs are present
3. Check that the bottom right tile coordinate is roughly equal to the corresponding WSI
   dimensions
4. Check that present tiles form a regular grid (equal column coords for all row coords)
5. Check that, for every mask variant, there is 100% correspondence with the scan tiles
   and mask tiles
"""

import argparse
from pathlib import Path
import sys

import cv2
import pandas as pd
from tqdm import tqdm

from presence_of_downscaled_scans import check_equivalence
import utils


common_cols = ["slide", "scanner"]


def find_scanner_paths(dataset_path, dataset):
    scanner_paths = [p for p in dataset_path.iterdir() if p.is_dir()]
    return scanner_paths


def get_scanner(path, dataset):
    scanner = utils.normalise_scanner(path.name)
    return scanner


def find_expected_tile_cases(dataset_path, extension, scanners):
    """
    Return a df with fields

    slide, scanner, path, height, width

    This is based on the scan tiles, and the size is based on the greatest coordinate
    """
    print(f"Discover scan tiles by traversing {dataset_path}")
    data = []
    dataset = dataset_path.name.lower()
    print(f"Dataset: {dataset}")
    scanner_paths = find_scanner_paths(dataset_path, dataset)
    for scanner_path in scanner_paths:
        scanner = get_scanner(scanner_path, dataset)
        if scanner not in scanners:
            print(f"skipping {scanner}")
            continue
        print(f"scanner: {scanner}")
        slide_paths = [p for p in scanner_path.iterdir() if p.is_dir()]
        for slide_path in tqdm(slide_paths):
            slide = slide_path.name
            tile_path = slide_path.joinpath(slide)
            if not tile_path.is_dir():
                print(f"ERROR: Could not find tile path {tile_path}")
                sys.exit()
            height, width = utils.find_dimensions_from_tiles(tile_path, extension)
            if height is not None and width is not None:
                data.append(
                    {
                        "slide": slide,
                        "scanner": scanner,
                        "path": slide_path,
                        "height": height,
                        "width": width,
                    }
                )
    df = pd.DataFrame(data)
    df = df.sort_values(by=common_cols)
    df = df.reset_index(drop=True)
    for scanner in scanners:
        num_found = len(df[df["scanner"] == "scanner"])
        print(f"Found number of cases from '{scanner}' scans: {num_found:>5}")
    return df


def find_expected_wsi_cases(wsi_source, extension, scanners):
    """
    Return a df with fields

    slide, scanner, path, height, width

    This is based on downscaled scans
    """
    if wsi_source.suffix == ".csv":
        print(f"Discover downscaled WSIs by reading {wsi_source}")
        df = pd.read_csv(wsi_source)
        # df = df.rename(columns={k: k.title() for k in df.columns})
        df["scanner"] = df["scanner"].apply(utils.normalise_scanner)
        if "section" in df.columns and "slide" not in df.columns:
            df = df.rename(columns={"section": "slide"})
    else:
        print(f"Discover downscaled WSIs by traversing {wsi_source}")
        dataset_path = wsi_source
        assert dataset_path.is_dir()
        data = []
        dataset = dataset_path.name.lower()
        print(f"Dataset: {dataset}")
        scanner_paths = find_scanner_paths(dataset_path, dataset)
        for scanner_path in scanner_paths:
            scanner = get_scanner(scanner_path, dataset)
            if scanner not in scanners:
                print(f"Skipping {scanner}")
                continue
            print(f"Scanner: {scanner}")
            slide_paths = [p for p in scanner_path.iterdir() if p.is_dir()]
            for slide_path in tqdm(slide_paths):
                slide = slide_path.name
                wsi_path = slide_path.joinpath(slide + extension)
                if not wsi_path.is_file():
                    print(f"ERROR: Could not find WSI path {wsi_path}")
                    sys.exit()
                image_grayscale = cv2.imread(str(wsi_path), cv2.IMREAD_GRAYSCALE)
                height, width = image_grayscale.shape
                data.append(
                    {
                        "slide": slide,
                        "scanner": scanner,
                        "height": height,
                        "width": width,
                    }
                )
        df = pd.DataFrame(data)
    df = df.sort_values(by=common_cols)
    df = df.reset_index(drop=True)
    for scanner in scanners:
        num_found = len(df[df["scanner"] == "scanner"])
        print(f"Found number of cases from '{scanner}' scans: {num_found:>5}")
    return df


def check_dimensions(tile_df, wsi_df, factor):
    tol = 5.0
    wsi_df["height"] = wsi_df["height"].apply(lambda x: x * factor)
    wsi_df["width"] = wsi_df["width"].apply(lambda x: x * factor)
    df = tile_df.merge(wsi_df, on=common_cols, how="inner")
    df["height-diff"] = df.apply(lambda r: r["height_x"] - r["height_y"], axis=1)
    df["width-diff"] = df.apply(lambda r: r["width_x"] - r["width_y"], axis=1)
    df["accept-diff"] = df.apply(
        lambda r: abs(r["height-diff"]) < tol and abs(r["width-diff"]) < tol, axis=1
    )
    ok = df["accept-diff"].values.all()
    utils.print_test_result(
        f"Difference in scan size from tiles and WSI are within tolerance {tol}", ok
    )
    if not ok:
        print(df[~df["accept-diff"]])


def check_tile_mesh(df, extension):
    df["tile-folder"] = df.apply(lambda r: r["path"].joinpath(r["slide"]), axis=1)
    df["complete-mesh"] = df["tile-folder"].apply(
        lambda p: utils.are_tiles_present(p, extension)
    )
    ok = df["complete-mesh"].values.all()
    utils.print_test_result("Tile folders are complete", ok)
    if not ok:
        print(df[~df["complete-mesh"]])


def are_tile_folders_corresponding(image_folder, mask_folder, mask_appendix, extension):
    if not image_folder.exists():
        print(f"WARNING: Image tile folder does not exist: {image_folder}")
        return False
    if not mask_folder.exists():
        print(f"WARNING: Mask tile folder does not exist: {mask_folder}")
        return False
    image_tiles = [p.stem for p in image_folder.glob(f"*{extension}")]
    mask_tiles = [p.stem.replace(mask_appendix, "") for p in mask_folder.glob("*.png")]
    return set(image_tiles) == set(mask_tiles)


def check_input_tiles(df, input_df, extension):
    df["tile-folder"] = df.apply(lambda r: r["path"].joinpath(r["slide"]), axis=1)
    tile_data = []
    for tile_folder in df["tile-folder"].values:
        for tile_path in tile_folder.glob(f"*{extension}"):
            tile_data.append({"ImagePath": str(tile_path)})
    tile_df = pd.DataFrame(tile_data)
    tile_df = tile_df.sort_values(by="ImagePath")
    tile_df = tile_df.reset_index(drop=True)
    input_df = input_df[["ImagePath"]]
    input_df["ImagePath"] = input_df["ImagePath"].apply(str)
    input_df = input_df.sort_values(by="ImagePath")
    input_df = input_df.reset_index(drop=True)
    check_equivalence(tile_df, input_df)


def check_corresponding_tiles(df, mask_variants, extension):
    df["scan-tile-folder"] = df.apply(lambda r: r["path"].joinpath(r["slide"]), axis=1)
    orig_columns = df.columns
    for mask_variant in mask_variants:
        df["mask-tile-folder"] = df["tile-folder"].apply(
            lambda p: p.with_name(p.name + mask_variant)
        )
        df["Corresponds"] = df.apply(
            lambda r: are_tile_folders_corresponding(
                r["scan-tile-folder"],
                r["mask-tile-folder"],
                mask_variant,
                extension,
            ),
            axis=1,
        )
        ok = df["Corresponds"].values.all()
        utils.print_test_result(
            f"Scan tiles correspond with tiles from mask variant {mask_variant}", ok
        )
        if not ok:
            print(df[~df["Corresponds"]])
        df = df[orig_columns]


def check_cli(args):
    tile_root = Path(args.tile_root)
    wsi_source = Path(args.wsi_source)
    input_csv_path = None if args.input_csv is None else Path(args.input_csv)
    scan_csv_path = None if args.scan_csv is None else Path(args.scan_csv)
    factor = float(args.factor)

    ok = True
    if not tile_root.is_dir():
        print(f"ERROR: Tile root must be directory: {tile_root}")
        ok = False
    name = tile_root.name.lower()
    if name.startswith("paip_"):
        name = name.replace("_", "-")

    if wsi_source.is_dir() and wsi_source.parts[-2:] == ("complete", "mpp-05_images"):
        wsi_source = wsi_source.joinpath(f"image-content_{name}.csv")
        if not wsi_source.exists():
            print(f"ERROR: Inferred wsi_source does not exist: {wsi_source}")
            ok = False

    if scan_csv_path is not None:
        if scan_csv_path.is_file():
            if scan_csv_path.suffix != ".csv":
                print(f"ERROR: Scan csv path must be csv file: {scan_csv_path}")
                ok = False
        else:
            scan_csv_path = scan_csv_path.joinpath(f"{name}.csv")
            if not wsi_source.exists():
                print(f"ERROR: Inferred scan_csv path does not exist: {scan_csv_path}")
                ok = False

    if input_csv_path is not None:
        if not input_csv_path.exists():
            print(f"ERROR: Input csv path does not exist: {input_csv_path}")
            ok = False
        if input_csv_path.is_dir():
            if args.resolution is None:
                print("ERROR: Input resolution is needed for inferring input csv name")
                ok = False
            if args.size is None:
                print("ERROR: Input size is needed for inferring input csv name")
                ok = False
            if args.overlap is None:
                print("ERROR: Input overlap is needed for inferring input csv name")
                ok = False
            if not ok:
                sys.exit()
            mpp = int(args.resolution)
            size = int(args.size)
            overl = int(args.overlap)
            input_csv_path = input_csv_path.joinpath(
                f"{name}_mpp-{mpp:02d}_tiles-{size:04d}_overl-{overl:04d}.csv"
            )
            if not input_csv_path.exists():
                print(f"ERROR: Inferred input_csv path does'nt exist: {input_csv_path}")
                ok = False

    if not (1 <= factor <= 100):
        print("ERROR: Factor is expected to be in range [1, 100]")
        ok = False
    if not ok:
        sys.exit()

    return tile_root, wsi_source, input_csv_path, scan_csv_path, factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tile_root", metavar="PATH", help="Root path of tiles for a single dataset"
    )
    parser.add_argument(
        "wsi_source",
        metavar="PATH",
        help=(
            "Csv with scan dimensions or root path of downscaled WSIs for a single "
            "dataset. Or folder with csv file which name is inferred from tile_root"
        ),
    )
    parser.add_argument(
        "factor",
        metavar="FLOAT",
        type=float,
        help=(
            "Downscale factor between wsi and tiles. E.g. if WSIs are mpp 5 and tiles "
            "are mpp 1, factor should be 5"
        ),
    )
    parser.add_argument(
        "--input_csv",
        metavar="PATH",
        help=(
            "Csv with input paths to tile or folder with csv file which name "
            "is inferred from tile_root, resolution, tile size, and overlap size"
        ),
    )
    parser.add_argument(
        "--scan_csv",
        metavar="PATH",
        help=(
            "Csv with expected scans to include, or folder with csv file which name "
            "is inferred from tile_root"
        ),
    )
    parser.add_argument(
        "--mask_variants",
        nargs="+",
        metavar="STR",
        choices=["foreground", "annotation", "foreground-annotation"],
        help="What mask tiles to look for, if any",
    )
    parser.add_argument(
        "--scan_extension",
        metavar="STR",
        choices=[".png", ".jpg"],
        default=".jpg",
        help="Scan tile file extension",
    )
    parser.add_argument(
        "--scanners",
        required=True,
        metavar="STR",
        nargs="+",
        choices=["aperio", "xr", "mirax", "mbm", "s210", "scanner"],
        help="Only check files from this scanner",
    )
    parser.add_argument(
        "--resolution",
        metavar="INT",
        type=int,
        help="If --input_csv is folder, give resolution in mpp",
    )
    parser.add_argument(
        "--size",
        metavar="INT",
        type=int,
        help="If --input_csv is folder, give tile size in pixels",
    )
    parser.add_argument(
        "--overlap",
        metavar="INT",
        type=int,
        help="If --input_csv is folder, give tile overlap size in pixels",
    )
    args = parser.parse_args()
    tile_root, wsi_source, input_csv_path, scan_csv_path, factor = check_cli(args)
    scan_tile_extension = args.scan_extension
    if args.mask_variants is None:
        mask_variants = None
    else:
        mask_variants = ["_mask-" + v for v in args.mask_variants]

    if scan_csv_path is not None:
        scan_df = pd.read_csv(scan_csv_path)
        scan_df = scan_df[common_cols]

    print("Command-line-input is ok. Start program")
    print()
    tile_df = find_expected_tile_cases(tile_root, scan_tile_extension, args.scanners)
    if scan_csv_path is not None:
        print(
            f"Check fields {common_cols} between existing tile folders and expected "
            "scan dataframe"
        )
        check_equivalence(tile_df[common_cols], scan_df)

    if input_csv_path is not None:
        print()
        input_df = pd.read_csv(input_csv_path)
        tile_df.to_csv("/tmp/tile_df.csv")
        print("Check input file")
        check_input_tiles(tile_df, input_df, scan_tile_extension)

    print()
    wsi_df = find_expected_wsi_cases(wsi_source, ".png", args.scanners)
    if scan_csv_path is not None:
        print(
            f"Check fields {common_cols} between existing downscaled WSIs and expected "
            "scan dataframe"
        )
        check_equivalence(wsi_df[common_cols], scan_df)

    if scan_csv_path is None:
        print()
        print(
            f"Check fields {common_cols} between existing tile folders and expected "
            "downscaled WSIs"
        )
        check_equivalence(tile_df[common_cols], wsi_df[common_cols])

    print()
    check_dimensions(tile_df, wsi_df, factor)
    check_tile_mesh(tile_df, scan_tile_extension)
    if mask_variants is not None:
        check_corresponding_tiles(tile_df, mask_variants, scan_tile_extension)
    else:
        print("No corresponding masks to check")


if __name__ == "__main__":
    main()
