import argparse
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import List, Mapping, Optional, Set, Sequence

import cv2
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import common_utils
from verify import utils as verify_utils

log = logging.getLogger()


def print_test_result(name: str, ok: bool):

    OKGREEN = "\033[92m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"

    msg = f"{OKGREEN}OK" if ok else f"{FAIL}FAIL"
    log.debug(f"{name:.<80}: {msg}{ENDC}")


class Config:
    def __init__(self, verbose: bool, overwrite: bool, gpu: Optional[int]):
        self.verbose = verbose
        self.overwrite = overwrite
        self.gpu = gpu  # None if cpu, else gpu device. For running neural network
        self.merge_threads = 10

        self.delete_scan_tiles = False
        self.delete_downsampled_scan = False
        self.delete_inference_tiles = False


class Paths:
    def __init__(
        self,
        scan_path_orig: Path,
        output_root: Path,
        model_path: Path,
        cache_root: Path,
    ):
        self.scan_path_orig = scan_path_orig
        self.scan_name = scan_path_orig.stem.replace(".", "-")
        if "." in scan_path_orig.stem:
            log.warning(
                "This program cannot handle '.' in file stems. File name\n"
                f"'{scan_path_orig.stem}'\n"
                "is replaced with\n"
                f"'{self.scan_name}'"
            )
        scan_suffix = scan_path_orig.suffix
        self.model_path = model_path
        self.src_root = Path(__file__).resolve().parent

        output_dir = output_root.joinpath(self.scan_name)
        self.output_dir = output_dir
        self.scan_cache = cache_root.joinpath(
            "scan_cache", f"{self.scan_name}{scan_suffix}"
        )
        self.scan_tiles = output_dir.joinpath("scan_tiles")
        self.scan_tile_list = self.scan_tiles.joinpath("tile_paths.csv")
        self.downsampled_scan = output_dir.joinpath("downsampled_scan")
        self.tile_inference = output_dir.joinpath("tile_inference")
        self.raw_probability = output_dir.joinpath("result_probability_raw.png")
        self.smooth_probability = output_dir.joinpath("result_probability_smooth.png")
        self.segmentation = output_dir.joinpath("result_segmentation.png")

        self.inference_config = self.src_root.joinpath("process/config/inference.toml")


def check_scan_file(path: Path) -> bool:
    extensions = [".svs", ".ndpi", ".mrxs", ".tiff"]
    ok = True
    if not path.is_file():
        log.error(f"Scan does not exist: {path}")
        ok = False
    if path.suffix not in extensions:
        log.error(f"Unsupported scan extension: {path.suffix} not in {extensions}")
        ok = False
    return ok


def check_cli(args: argparse.Namespace) -> List[Path]:
    if len(args.input) == 1:
        input_path = args.input[0]
        if input_path.suffix == ".csv":
            df = pl.read_csv(input_path)
            if "Path" not in df.columns:
                log.error("Expected 'Path' header in input .csv file:\n{df}")
                exit()
            scan_paths = df.get_column("Path").sort().to_list()
        else:
            if not check_scan_file(input_path):
                exit()
            else:
                scan_paths = [input_path]
    else:
        scan_paths = []
        for input_path in args.input:
            if check_scan_file(input_path):
                scan_paths.append(input_path)

    if not args.model.is_file():
        log.error(f"Model path is not a file: {args.model}")
        exit()
    if not args.model.suffix == ".tar":
        log.error(f"Model path is not a tar file: {args.model}")
        exit()

    if args.gpu is not None and args.gpu < 0:
        log.error(f"--gpu value must be a non-negative integer, got {args.gpu}")
        exit()

    return scan_paths


def create_input_tile_list(paths: Paths):
    tile_folder = paths.scan_tiles.joinpath(paths.scan_name)
    df = pl.DataFrame({"ImagePath": sorted([str(p) for p in tile_folder.glob("*jpg")])})
    df.write_csv(paths.scan_tile_list)


def run_command(
    command: Sequence[str],
    verbose: bool,
    env: Optional[Mapping[str, str]] = None,
):
    log.debug(f"Command:\n'{' '.join(command)}'")
    if verbose:
        if env is None:
            subprocess.check_call(command)
        else:
            subprocess.check_call(command, env=env)
    else:
        if env is None:
            subprocess.check_call(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            subprocess.check_call(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
            )


def copy_scan(paths: Paths):
    if not paths.scan_cache.exists():
        log.info(f"Copy to '{paths.scan_cache}'")
        paths.scan_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(paths.scan_path_orig, paths.scan_cache)


def check_cache(paths: Paths):
    if paths.scan_cache.exists():
        scan_path = paths.scan_cache
        cache = False
    else:
        scan_path = paths.scan_path_orig
        cache = True
    return scan_path, cache


def tile_scan(paths: Paths, config: Config):
    scan_path, cache = check_cache(paths)
    log.info(f"Create tiles from '{scan_path}'")
    try:
        command = [
            "python3",
            str(paths.src_root.joinpath("preprocess/scan_tiling/tile_scans.py")),
            str(scan_path),
            "--output",
            str(paths.scan_tiles),
            "--format",
            "jpg",
            "--mpp",
            "1",
            "--mode",
            "overlap",
            "--size",
            "7680",
            "--min_overlap",
            "1024",
            "--overwrite",
            "yes" if config.overwrite else "no",
        ]
        if cache:
            command += ["--cache", str(paths.scan_tiles)]
        run_command(command, config.verbose)
    except subprocess.CalledProcessError as err:
        raise ChildProcessError(err)
    else:
        create_input_tile_list(paths)


def downsample_scan(paths: Paths, config: Config):
    scan_path, cache = check_cache(paths)
    log.info(f"Downsample '{scan_path}'")
    try:
        command = [
            "python3",
            str(paths.src_root.joinpath("preprocess/prepare_scans/prepare_scans.py")),
            "--scans",
            str(scan_path),
            "--output",
            str(paths.downsampled_scan),
            "--mask",
            "foreground",
            "--mpp",
            "5",
            "--overwrite",
            "yes" if config.overwrite else "no",
        ]
        if cache:
            command += ["--cache", str(paths.downsampled_scan)]
        run_command(command, config.verbose)
    except subprocess.CalledProcessError as err:
        raise ChildProcessError(err)


def check_path_existence(name: str, path: Path, ok: bool) -> bool:
    if not path.exists():
        log.error(f"{name} does not exist: {path}")
        ok = False
    print_test_result(f"Check exists: {name}", ok)
    return ok


def check_int(name: str, v1: int, v2: int, ok: bool, tolerance: int = 0) -> bool:
    if tolerance == 0:
        check = v1 == v2
        error_message = f"{name} unequal: {v1} != {v2}"
    else:
        check = abs(v1 - v2) < tolerance
        error_message = f"{name} not within tolerance: abs({v1} - {v2}) >= {tolerance}"
    if not check:
        log.error(error_message)
        ok = False
    print_test_result(f"Check int: {name}", ok)
    return ok


def check_sets(name1: str, name2: str, set1: Set, set2: Set, ok: bool) -> bool:
    if not set1 == set2:
        ok = False
        log.error(f"Sets are unequal: '{name1}' != '{name2}':")
        missing_1 = set2.difference(set1)
        log.error(f"Elements in '{name2}' not in '{name1}': {len(missing_1)}")
        for element in sorted(list(missing_1)):
            log.error(element)
        missing_2 = set1.difference(set2)
        log.error(f"Elements in '{name1}' not in '{name2}': {len(missing_2)}")
        for element in sorted(list(missing_2)):
            log.error(element)
    print_test_result(f"Check sets: {name1} == {name2}", ok)
    return ok


def check_tile_existence(tile_folder: Path, tile_list_path: Path, ok: bool) -> bool:
    existing = set([p.name for p in tile_folder.glob("*.jpg")])
    # Compare with 'expected_tiles.txt' which should be created during tiling
    expected_path = tile_folder.joinpath("expected_tiles.txt")
    ok = check_path_existence("Expected tiles file", expected_path, ok)
    if ok:
        with expected_path.open() as f:
            expected = set([r.strip() for r in f.readlines()])
        ok = check_sets("existing tiles", "expected tiles", existing, expected, ok)
    # Compare with 'tile_paths.csv' which should be created after tiling
    ok = check_path_existence("Tile list file", tile_list_path, ok)
    if ok:
        input_paths = pl.read_csv(tile_list_path).get_column("ImagePath").to_list()
        input_names = set([Path(p).name for p in input_paths])
        ok = check_sets("existing tiles", "input list tiles", existing, input_names, ok)
    return ok


def check_tile_grid(tile_folder: Path, ok: bool):
    if not verify_utils.are_tiles_present(tile_folder, ".jpg"):
        log.error(f"Tiles in tile folder does not form a complete grid: {tile_folder}")
        ok = False
    print_test_result(f"Check tile grid: {tile_folder.name}", ok)
    return ok


def verify_scan_tiling(paths: Paths, verbose: bool):
    log.debug("Verify scan tiling")
    ok = True
    tol = 5
    factor = 5  # Tiles are 1 MPP, downsampled scans are 5 MPP

    # 1. Check that scan tiles are present
    tile_folder = paths.scan_tiles.joinpath(paths.scan_name)
    tile_list_path = paths.scan_tile_list
    ok = check_path_existence("Scan tile folder", tile_folder, ok)
    ok = check_tile_existence(tile_folder, tile_list_path, ok)

    # 2. Check that downsampled scan and associated files are present
    downsampled_scan = paths.downsampled_scan.joinpath(paths.scan_name + ".png")
    background_mask = paths.downsampled_scan.joinpath(
        paths.scan_name + "_mask-foreground.png"
    )
    ok = check_path_existence("Downsampled scan", downsampled_scan, ok)
    ok = check_path_existence("Background mask", background_mask, ok)

    # 3. Check that merged tile dimensions roughly match the downsampled scan dimensions
    height_tiles, width_tiles = verify_utils.find_dimensions_from_tiles(
        tile_folder, ".jpg"
    )
    height_tiles = int(height_tiles / factor)
    width_tiles = int(width_tiles / factor)
    image_grayscale = cv2.imread(str(downsampled_scan), cv2.IMREAD_GRAYSCALE)
    mask_grayscale = cv2.imread(str(background_mask), cv2.IMREAD_GRAYSCALE)
    height_scan, width_scan = image_grayscale.shape
    ok = check_int("Height scan", height_scan, mask_grayscale.shape[0], ok)
    ok = check_int("Width scan", width_scan, mask_grayscale.shape[1], ok)
    ok = check_int("Height merged tiles", height_scan, height_tiles, ok, tol)
    ok = check_int("Width merged tiles", width_scan, width_tiles, ok, tol)

    # 4. Check that present tiles form a regular grid
    ok = check_tile_grid(tile_folder, ok)

    if not ok:
        raise ValueError("Failed scan tile verification")


def tile_inference(paths, config):
    log.info(f"Create probability tiles using '{paths.model_path}'")
    try:
        # paths.tile_inference.mkdir(exist_ok=True)
        command = [
            "python3",
            str(paths.src_root.joinpath("segment_images.py")),
            "--config",
            str(paths.inference_config),
            "--input",
            str(paths.scan_tile_list),
            "--full_image",
            str(paths.downsampled_scan),
            "--restore",
            str(paths.model_path),
            "--run_dir",
            str(paths.tile_inference),
            "--overwrite",
            "yes" if config.overwrite else "no",
        ]
        if config.gpu is not None:
            command.extend(["--gpu", str(config.gpu)])
        run_command(command, config.verbose)
    except subprocess.CalledProcessError as err:
        raise ChildProcessError(err)


def compare_file_collections(
    root_1: Path,
    root_2: Path,
    extension_1: str,
    extension_2: str,
    ok: bool,
    exclude_pattern: Optional[str] = None,
):
    if not root_1.is_dir():
        log.error(f"Root folder is not a directory: {root_1}")
        ok = False
        return ok
    if not root_2.is_dir():
        log.error(f"Root folder is not a directory: {root_2}")
        ok = False
        return ok

    paths_1 = list(root_1.rglob(f"*{extension_1}"))
    paths_2 = list(root_2.rglob(f"*{extension_2}"))
    if exclude_pattern is not None:
        paths_1 = [p for p in paths_1 if exclude_pattern not in p.stem]
        paths_2 = [p for p in paths_2 if exclude_pattern not in p.stem]

    names_1 = set([p.relative_to(root_1).with_suffix("") for p in paths_1])
    names_2 = set([p.relative_to(root_2).with_suffix("") for p in paths_2])

    ok = check_sets(
        f"{extension_1} files in {root_1.name}",
        f"{extension_2} files in {root_2.name}",
        names_1,
        names_2,
        ok,
    )

    return ok


def verify_tile_inference(paths: Paths, verbose: bool):
    log.debug("Verify tile inference")
    ok = True
    ok = compare_file_collections(
        paths.scan_tiles,
        paths.tile_inference.joinpath("probability_maps_class-0"),
        ".jpg",
        ".png",
        ok,
    )
    ok = compare_file_collections(
        paths.scan_tiles,
        paths.tile_inference.joinpath("probability_maps_class-127"),
        ".jpg",
        ".png",
        ok,
    )
    ok = compare_file_collections(
        paths.scan_tiles,
        paths.tile_inference.joinpath("probability_maps_class-255"),
        ".jpg",
        ".png",
        ok,
    )

    if not ok:
        raise ValueError("Failed scan tile verification")


def merge_inference_tiles(paths: Paths, config: Config):
    # NOTE: We copy the tile folder of interest to a new name
    # Folder from tile inference: `probability_maps_class-255
    # Expected folder in merging: '<scan name>' which is also common tile name prefix
    # Symlinking did not work (docker issue?). Renaming does work, but the tile
    # inference program looks for a folder with the name 'probability_maps_class-255'
    # and retiles if it does not exist.
    # Copying should be cheap, so this is done until something better is implemented
    # TODO: This results in a duplication which might be confusion, and it is currently
    # solved by deleting 'existing' before renaming 'expected' to 'existing'
    existing = paths.tile_inference.joinpath("probability_maps_class-255")
    expected = paths.tile_inference.joinpath(paths.scan_name)
    if not expected.exists():
        shutil.copytree(existing, expected)
    # Extra verification step
    ok = compare_file_collections(
        paths.scan_tiles,
        expected,
        ".jpg",
        ".png",
        True,
    )
    if not ok:
        raise ValueError(f"In expected tile folder: {expected}")
    log.info(f"Merge tiles in '{expected}'")

    factor = 5  # Tiles are 1 MPP, downsampled scans are 5 MPP
    program_path = str(
        paths.src_root.joinpath(
            "preprocess/tile_with_overlap/target/release/tile_with_overlap"
        )
    )
    try:
        env = {"RAYON_NUM_THREADS": f"{config.merge_threads}"}
        command = [
            program_path,
            "--input",
            str(expected),
            "--output",
            str(paths.raw_probability),
            "--reference",
            str(paths.downsampled_scan),
            "--reference_factor",
            str(factor),
            "--restore",
            "distance",
        ]
        run_command(command, config.verbose, env)
        shutil.rmtree(existing)
        os.rename(expected, existing)
    except subprocess.CalledProcessError as err:
        raise ChildProcessError(err)


def segment_probability(paths: Paths, config: Config):
    log.info(f"Segment probability map '{paths.raw_probability}'")
    downsampled_scan = paths.downsampled_scan.joinpath(paths.scan_name + ".png")
    try:
        command = [
            "python3",
            str(paths.src_root.joinpath("postprocess/segment_probability_maps.py")),
            str(paths.raw_probability),
            "--images",
            str(downsampled_scan),
            "--output_mask",
            str(paths.segmentation),
            "--output_prob",
            str(paths.smooth_probability),
            "--smooth",
            "--method",
            "hysteresis-85-229",
            "--prune",
            "percentile-95-229",
            "--overwrite",
            "yes" if config.overwrite else "no",
        ]
        run_command(command, config.verbose)
    except subprocess.CalledProcessError as err:
        raise ChildProcessError(err)
    log.info(f"Segmentation result: '{paths.segmentation}'")


def verify_segmentation(paths: Paths, verbose: bool):
    log.debug("Verify segmentation")
    ok = True
    ok = check_path_existence("Raw probability", paths.raw_probability, ok)
    ok = check_path_existence("Smooth probability", paths.smooth_probability, ok)
    ok = check_path_existence("Final segmentation", paths.segmentation, ok)
    if not ok:
        raise ValueError("Failed final segmentation verification")


def remove_folder(path: Path):
    log.debug(f"Removing '{path}'")
    if not path.exists():
        log.warning(f"Removing non-existing folder: {path}")
    else:
        shutil.rmtree(path)


def clean_up(paths: Paths, config: Config):
    log.info("Cleaning up")
    if paths.scan_cache.exists():
        remove_folder(paths.scan_cache.parent)
    if config.delete_scan_tiles:
        remove_folder(paths.scan_tiles)
    if config.delete_inference_tiles:
        remove_folder(paths.tile_inference)
    if config.delete_downsampled_scan:
        remove_folder(paths.downsampled_scan)
    if paths.output_dir.joinpath("postprocess.log").exists():
        os.unlink(paths.output_dir.joinpath("postprocess.log"))


def run_pipeline(paths: Paths, config: Config):
    log.info("")
    log.info(f"Process '{paths.scan_path_orig}'")
    copy_scan(paths)
    tile_scan(paths, config)
    downsample_scan(paths, config)
    verify_scan_tiling(paths, config.verbose)
    tile_inference(paths, config)
    verify_tile_inference(paths, config.verbose)
    merge_inference_tiles(paths, config)
    segment_probability(paths, config)
    verify_segmentation(paths, config.verbose)
    clean_up(paths, config)


def main():
    parser = argparse.ArgumentParser(
        description="Scan segmentation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input",
        metavar="PATH",
        type=Path,
        nargs="+",
        help=(
            "Accepted input:\n"
            " - Single .csv with scan paths and 'Path' as header\n"
            " - Scan path(s)\n"
        ),
    )
    parser.add_argument(
        "output",
        metavar="PATH",
        type=Path,
        help="Output root folder. One folder per input scan will be created below this",
    )
    parser.add_argument(
        "cache",
        metavar="PATH",
        type=Path,
        help="Path to scan cache",
    )
    parser.add_argument(
        "model",
        metavar="PATH",
        type=Path,
        help="Path to model (.tar)",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        metavar="INT",
        type=int,
        default=0,
        help="Which gpu to use when applying a segmentation model",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results, or keep them (default)",
    )
    args = parser.parse_args()

    start_time = time.time()
    common_utils.setup_logging(args.verbose)
    log.info("Start segmentation pipeline")

    scan_paths = check_cli(args)
    log.info(
        f"Found {len(scan_paths)} input scan{common_utils.suffix(len(scan_paths))}"
    )
    for scan_path_orig in scan_paths:
        log.debug(f"Segmenting {scan_path_orig.name}")
        paths = Paths(scan_path_orig, args.output, args.model, args.cache)
        config = Config(args.verbose, args.overwrite, args.gpu)
        if paths.segmentation.exists():
            log.warning(f"Segmentation result exist: {paths.segmentation}")
            if args.overwrite:
                log.warning("Reprocess scan and overwrite results")
            else:
                log.warning("Skip scan. Pass --overwrite to reprocess")
                continue
        try:
            run_pipeline(paths, config)
        except Exception as e:
            log.error(e)

    log.info("Segmentation pipeline finished.")
    log.info(f"Elapsed time: {common_utils.format_time(time.time() - start_time)}")


if __name__ == "__main__":
    main()
