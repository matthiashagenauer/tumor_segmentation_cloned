import sys
import argparse
from pathlib import Path
import logging
import time
from typing import List, Optional, Sequence

from corresponding import Case, find_corresponding_paths
import configuration
from common import common_utils
import parallel_tiling

log = logging.getLogger()


def check_cli(args: argparse.Namespace):
    ok = True
    for p in args.scans:
        if not p.exists():
            log.error(f"Path does not exist: {p}")
            ok = False
    if args.min_overlap is not None and args.mode != "overlap":
        log.error("Using --min_overlap requires --mode='overlap'")
        ok = False

    if not ok:
        log.error("Terminating because of erroneous input argument values")
        sys.exit()


def process_cli(args: argparse.Namespace) -> argparse.Namespace:
    args.scans = [Path(p) for p in args.scans]
    args.output = common_utils.maybe_path(args.output)
    args.cache = common_utils.maybe_path(args.cache)

    check_cli(args)
    return args


def filter_existing(samples: Sequence[Case], overwrite: Optional[bool]) -> List[Case]:
    # We only care about checking for existing tiled scans (and not annotations)
    log.info("Check existing scan output tiles")
    include_samples = []
    num_existing = 0
    tile_formats = configuration.valid_tile_formats()
    for sample in samples:
        # Only perform check if this file exist, otherwise it is difficult to know which
        # tiles should be present
        include = True
        if sample.scan_output.joinpath("expected_tiles.txt").exists():
            with sample.scan_output.joinpath("expected_tiles.txt").open() as f:
                expected = set([r.strip() for r in f.readlines()])
            existing = set(
                [p.name for e in tile_formats for p in sample.scan_output.glob(f"*{e}")]
            )
            if expected == existing:
                num_existing += 1
                if overwrite is None:
                    log.error(f"Output scan tile folder exist: {sample.scan_output}")
                    log.error("Decide what to do with --overwrite option")
                    sys.exit()
                else:
                    include = overwrite
        if include:
            include_samples.append(sample)
    diff = len(samples) - len(include_samples)
    if diff != 0:
        assert overwrite is False
        log.info(f"Excluding {diff} samples with existing output")
    if overwrite:
        log.info(f"Overwriting {num_existing} samples with existing output")
    return include_samples


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "scans",
        metavar="PATH",
        nargs="+",
        help="List of scan paths or root folder with scans below",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Output root folder. Dry run if not given.",
    )
    parser.add_argument(
        "--masks",
        metavar="PATH",
        help=(
            "Root folder where masks (png) reside. Only include tiles that fall "
            "inside the region (for some configurable definition) set by the mask.\n"
            "scan:      <--scans>/relative/path/to/<stem>.<extension>\n"
            ".png mask: <--masks>/relative/path/to/<formatted><mask_appendix>.png\n"
        ),
    )
    parser.add_argument(
        "--mask_appendix",
        metavar="STR",
        default="",
        help=(
            "If --masks is given, use this to locate the mask file if it is stored "
            "together with an image file."
        ),
    )
    parser.add_argument(
        "--format",
        metavar="STR",
        choices=["png", ".png", "jpg", ".jpg", "jpeg", ".jpeg"],
        default="jpg",
        help=(
            "Image format to store result scan tiles. [default: %(default)s]"
            "Annotation tiles are stored as png"
        ),
    )
    parser.add_argument(
        "--mpp",
        metavar="FLOAT",
        type=float,
        default=1.0,
        help="Target resolution in micrometers per pixel. [default: %(default)f].",
    )
    parser.add_argument(
        "--scanners",
        metavar="STR",
        nargs="+",
        choices=list(common_utils.supported_scanners.keys()),
        help=(
            "Expected scanners to look for. "
            f"Choices {list(common_utils.supported_scanners.keys())}"
        ),
    )
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
        choices=["overlap", "inside", "outside", "rest"],
        default="inside",
        help=(
            """Determine how to tile:
- 'overlap': The scan is tiled from the top left corner with overlap (if
  required) to ensure that the entire scan is covered. The overlap is
  approximately equal between all adjacent tiles. Minimum allowed overlap
  can be specified.
- 'inside': The scan is tiled from the top left corner without overlap. The
  last tile is the last tile that fully fit inside the scan in either
  direction.
- 'outside': The scan is tiled from the top left corner without overlap. The
  last tile is the first non-filled tile in either direction, except when
  all tiles fit perfectly in the scan.
- 'rest': The scan is tiled from the top left corner without overlap. If the
  tile grid does not match perfectly the scan in one direction, the last set
  of tiles in this direction will be smaller than the target size in this
  direction. With this we have non-overlapping tiles where the leftmost and
  bottommost tiles are potentially smaller in width or height, respectively
  so that the tiles together cover the scan perfectly.
[default: %(default)s]"""
        ),
    )
    parser.add_argument(
        "--min_overlap",
        metavar="INT",
        type=int,
        help=(
            "Require at least this many overlapping pixels in both directions. \n"
            "Conflicts with --mode != 'overlap'."
        ),
    )
    parser.add_argument(
        "--cache",
        metavar="PATH",
        help=(
            "Copy scans to temporary local cache and load scans for tiling from there. "
            "Specify cache path here, otherwise no copying happens."
        ),
    )
    parser.add_argument(
        "--tile_workers",
        metavar="INT",
        type=int,
        default=40,
        help="Number of workers to use for tiling. [default: %(default)d]",
    )
    parser.add_argument(
        "--copy_workers",
        metavar="INT",
        type=int,
        default=5,
        help="Number of workers to use for copying. [default: %(default)d]",
    )
    parser.add_argument(
        "--add_scan_folder",
        action="store_true",
        help=(
            "If input scans are all in the same folder, the output will also be so if "
            " this is not set"
        ),
    )
    parser.add_argument(
        "--discard_zero_valued_tiles",
        action="store_true",
        help="Discard all tiles that only contain the value 0",
    )
    parser.add_argument(
        "--overwrite",
        metavar="STR",
        choices=["yes", "no"],
        help="If result file exist: overwrite ('yes') or skip ('no')",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args()
    start_time = time.time()

    common_utils.setup_logging(args.verbose)

    args = process_cli(args)
    conf = configuration.Configuration(args)
    if args.masks:
        conf.tile_inside_mask = True

    if args.scanners is None:
        scan_extensions = set(common_utils.supported_scanners.values())
    else:
        scan_extensions = common_utils.extensions_from_scanners(args.scanners)
    scan_paths, scan_root = common_utils.find_files(args.scans, scan_extensions)
    plural = common_utils.plural_s(len(scan_paths))
    log.info(f"Found {len(scan_paths)} input scan file{plural}")

    annotation_paths = None
    annotation_root = None

    if args.output is None:
        log.info("Output path is not given, terminating program")
        return

    samples = find_corresponding_paths(
        scan_paths,
        scan_root,
        annotation_paths,
        annotation_root,
        args.masks,
        args.mask_appendix,
        args.output,
        args.add_scan_folder,
    )
    samples = filter_existing(samples, conf.overwrite)

    if len(samples) == 0:
        log.error("No input scans to tile")
        return

    plural = common_utils.plural_s(len(samples))
    log.info(f"Processing {len(samples)} input scan file{plural}")
    if len(samples) == 0:
        log.error("Found 0 samples to tile")
    if conf.tile_inside_mask:
        new_samples = [s for s in samples if s.mask is not None]
        diff = len(samples) - len(new_samples)
        if diff > 0:
            log.info(f"Excluding {diff} samples without masks")
        samples = new_samples
    parallel_tiling.tile_cases(samples, conf)
    log.info("Program finished.")
    log.info(f"Elapsed time: {common_utils.format_time(time.time() - start_time)}")


if __name__ == "__main__":
    main()
