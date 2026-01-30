"""
Assumes reference tiles and full images are ok.

Check inference tiles and full images against reference
"""

import argparse
from pathlib import Path
import sys
from typing import Any, List, Optional, Sequence, Set, Tuple

import utils


class Config:
    def __init__(self):
        self.input_tile_extension = ".png"
        self.input_image_extension = ".png"
        self.ref_tile_extension = ".jpg"
        self.ref_image_extension = ".png"

        self.tile_mpp = 1
        self.image_mpp = 5
        self.tile_size = 7680
        self.tile_overlap = 1024
        self.checkpoint = 500000
        # self.segmentation = "argmax"
        self.segmentation = "smooth_hysteresis-85-229_percentile-95-229"
        self.scanners = ["AP", "Aperio", "APERIO", "XR", "MIRAX", "S210", "scanner"]


class Corresponding:
    def __init__(
        self,
        input_root: Path,
        appendix: str,
        ref_tile_root: Path,
        ref_image_root: Path,
        scanner: Optional[str],
        scanner_level: bool,
    ):
        conf = Config()
        self.dataset = input_root.name.replace(f"_{appendix}", "")
        if self.dataset.startswith("paip"):
            self.dataset_folder_name = "paip_" + "-".join(self.dataset.split("-")[1:])
        else:
            self.dataset_folder_name = self.dataset.upper()

        if scanner_level:
            assert scanner is not None
            scanner_from_parent = scanner
            std_scanner_from_parent = utils.normalise_scanner(scanner)
        else:
            scanner_from_parent = None
            std_scanner_from_parent = None

        if (
            self.dataset.endswith("-ap")
            or self.dataset.endswith("-aperio")
            or self.dataset.endswith("-xr")
            or self.dataset.endswith("-mirax")
            or self.dataset.endswith("-s210")
        ):
            scanner_from_dataset = self.dataset.split("-")[-1]
            self.dataset = self.dataset.replace("-" + scanner_from_dataset, "")
            std_scanner_from_dataset = utils.normalise_scanner(scanner_from_dataset)
        else:
            scanner_from_dataset = None
            std_scanner_from_dataset = None

        if scanner_from_parent is None and scanner_from_dataset is None:
            print(f"WARNING: Unable to infer scanner from {input_root}")
            if scanner is not None:
                print(f"Using input scanner {scanner}")
                self.scanner = scanner
            else:
                print("Try to input --scanner")
                sys.exit()
        elif scanner_from_parent is None and scanner_from_dataset is not None:
            self.scanner = std_scanner_from_dataset
        elif scanner_from_parent is not None and scanner_from_dataset is None:
            self.scanner = std_scanner_from_parent
        else:
            if std_scanner_from_parent != std_scanner_from_dataset:
                print(f"WARNING: Unequal assumed scanner from {self.dataset}:")
                print(f"{std_scanner_from_parent} != {std_scanner_from_dataset}")
            self.scanner = std_scanner_from_parent
        print(f"Assuming dataset name is {self.dataset}")
        print(f"Assuming scanner name is {self.scanner}")
        print(f"Finding corresponding paths for {self.dataset} {self.scanner}")

        # Inference
        result = input_root.joinpath(f"results/default/checkpoint_{conf.checkpoint}")
        assert result.is_dir(), f"Does not exist: {result}"
        self.prob_tile = result.joinpath("probability_maps_class-255")
        self.prob_image = result.joinpath("merged_probability_maps")
        self.seg_image = result.joinpath(f"segmentation_masks/{conf.segmentation}")
        if scanner_level:
            self.prob_tile = self.prob_tile.joinpath(scanner_from_parent)
            self.prob_image = self.prob_image.joinpath(scanner_from_parent)
            self.seg_image = self.seg_image.joinpath(scanner_from_parent)

        # Reference
        ref_tile_parent = ref_tile_root.joinpath(self.dataset_folder_name)
        ref_image_parent = ref_image_root.joinpath(self.dataset_folder_name)
        assert ref_tile_parent.is_dir(), f"Does not exist: {ref_tile_parent}"
        assert ref_image_parent.is_dir(), f"Does not exist: {ref_tile_parent}"

        if self.scanner == "aperio":
            names = set(["Aperio", "AP", "APERIO"])
        elif self.scanner == "xr":
            names = set(["XR"])
        elif self.scanner == "mirax":
            names = set(["MIRAX"])
        elif self.scanner == "s210":
            names = set(["S210"])
        elif self.scanner == "scanner":
            names = set(["scanner"])
        else:
            print(f"ERROR: Invalid scanner: '{scanner}'")
            sys.exit()
        subfolders = list(ref_tile_parent.iterdir())
        scanner_names = set([p.name for p in subfolders]).intersection(names)
        assert len(scanner_names) == 1, f"Expected 1 scanner, got {len(scanner_names)}"
        scanner_name = list(scanner_names)[0]
        assert scanner_name in conf.scanners, f"{scanner_name} not in {conf.scanners}"
        self.ref_tile = ref_tile_parent.joinpath(scanner_name)
        self.ref_image = ref_image_parent.joinpath(scanner_name)
        assert self.ref_tile.is_dir(), f"Does not exist: {self.ref_tile}"
        assert self.ref_image.is_dir(), f"Does not exist: {self.ref_image}"


def scanner_level(path: Path) -> bool:
    conf = Config()
    names = set([p.name for p in path.iterdir()])
    return names.difference(set(conf.scanners)) == set()


def find_corresponding(
    input_roots: Sequence[Path],
    ref_tile_root: Path,
    ref_image_root: Path,
    input_scanner: Optional[str],
) -> List[Corresponding]:
    conf = Config()

    collection: List[Corresponding] = []
    for input_root in input_roots:
        prob_tile = input_root.joinpath(
            f"results/default/checkpoint_{conf.checkpoint}/probability_maps_class-255"
        )
        if not prob_tile.is_dir():
            print(f"ERROR: Tile probability dir does not exist: {prob_tile}")
            sys.exit()
        if len(list(prob_tile.iterdir())) == 0:
            print(f"ERROR: Empty tile probability dir: {prob_tile}")
            sys.exit()
        appendix = (
            f"mpp-{conf.tile_mpp:02d}_"
            f"tiles-{conf.tile_size:04d}_"
            f"overl-{conf.tile_overlap:04d}"
        )
        if not input_root.name.endswith(appendix):
            print(f"WARNING: invalid appendix {appendix} in {input_root}")
            continue
        if scanner_level(prob_tile):
            scanners = sorted([p.name for p in prob_tile.iterdir()])
            for scanner in scanners:
                try:
                    collection.append(
                        Corresponding(
                            input_root,
                            appendix,
                            ref_tile_root,
                            ref_image_root,
                            scanner,
                            True,
                        )
                    )
                except Exception as err:
                    print(f"ERROR: Finding corresponding scanner paths: {input_root}")
                    print(err)
                    sys.exit()
        else:
            try:
                collection.append(
                    Corresponding(
                        input_root,
                        appendix,
                        ref_tile_root,
                        ref_image_root,
                        input_scanner,
                        False,
                    )
                )
            except Exception as err:
                print(f"ERROR: Finding corresponding image paths: {input_root}")
                print(err)
                sys.exit()
    return collection


def check_sets(pred: Set[Any], ref: Set[Any], verbose: bool):
    if len(pred) > 0:
        ok = len(pred) == len(ref)
    else:
        ok = False
    utils.print_test_result(f"{'':>8}Count", ok)
    if not ok and verbose:
        print(f"{'':>12}Predicted count: {len(pred)}")
        print(f"{'':>12}Reference count: {len(ref)}")

    missing_pred = ref.difference(pred)
    ok = len(missing_pred) == 0
    utils.print_test_result(f"{'':>8}Missing predictions", ok)
    if not ok and verbose:
        print(f"{'':>12}Number of missing: {len(missing_pred)}")

    missing_refs = pred.difference(ref)
    ok = len(missing_refs) == 0
    utils.print_test_result(f"{'':>8}Missing references", ok)
    if not ok and verbose:
        print(f"{'':>12}Number of missing: {len(missing_refs)}")
        # for p in sorted(list(pred.difference(ref))):
        #     print(p)


def check_images(
    input_root: Path,
    ref_root: Path,
    input_extension: str,
    ref_extension: str,
    verbose: bool,
):
    if not input_root.is_dir():
        print(f"{'':>8}Input root does not exist: {input_root}")
        return

    input_paths = list(input_root.rglob(f"*{input_extension}"))
    ref_paths = [
        p for p in ref_root.rglob(f"*{ref_extension}") if "_mask" not in p.stem
    ]

    input_images = set([p.relative_to(input_root).with_suffix("") for p in input_paths])
    ref_images = set([p.relative_to(ref_root).with_suffix("") for p in ref_paths])

    check_sets(input_images, ref_images, verbose)


def check_collection(coll: Corresponding, verbose: bool):
    conf = Config()
    print(f"{'':>4}Probability map tiles")
    check_images(
        coll.prob_tile,
        coll.ref_tile,
        conf.input_tile_extension,
        conf.ref_tile_extension,
        verbose,
    )
    print(f"{'':>4}Merged probability map images")
    check_images(
        coll.prob_image,
        coll.ref_image,
        conf.input_image_extension,
        conf.ref_image_extension,
        verbose,
    )
    print(f"{'':>4}Segmentation mask images")
    check_images(
        coll.seg_image,
        coll.ref_image,
        conf.input_image_extension,
        conf.ref_image_extension,
        verbose,
    )


def check_input(args: argparse.Namespace) -> Tuple[List[Path], Path, Path]:
    input_roots = [Path(p) for p in args.input]
    ref_tile_root = Path(args.tiles)
    ref_image_root = Path(args.images)

    assert all([p.is_dir() for p in input_roots]), "Input paths"

    assert ref_tile_root.is_dir(), f"Ref tile root is dir: {ref_tile_root}"
    if ref_tile_root.name != "tiled-scans_jpg_mpp-01_size-7680_overlap-1024":
        print("ERROR: Ref tile root name")
        print(
            f"{ref_tile_root.name} != 'tiled-scans_jpg_mpp-01_size-7680_overlap-1024'"
        )
        sys.exit()

    assert ref_image_root.is_dir(), f"Ref image root name: {ref_image_root}"
    assert ref_image_root.name == "downscaled-scans_png_mpp-05", "Ref image root name"

    return input_roots, ref_tile_root, ref_image_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", metavar="PATH", nargs="+", help="Single dataset inference result root"
    )
    parser.add_argument("tiles", metavar="PATH", help="Reference tile root folder")
    parser.add_argument("images", metavar="PATH", help="Reference image root folder")
    parser.add_argument(
        "--scanner",
        metavar="PATH",
        help="Specify scanner if it is not possible to infer from input",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print()
    input_roots, ref_tile_root, ref_image_root = check_input(args)

    collections = find_corresponding(
        input_roots, ref_tile_root, ref_image_root, args.scanner
    )

    for c in collections:
        print()
        print(c.dataset, c.scanner)
        check_collection(c, args.verbose)


if __name__ == "__main__":
    main()
