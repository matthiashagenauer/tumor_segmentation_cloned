"""
Prepare original data for use by the segmentation network

Ole-Johan Skrede
16.04.2021
"""
import argparse
import datetime
import logging
import os
from pathlib import Path
import shutil
import tempfile

import cv2
import json
from tqdm import tqdm
import pandas as pd

import utils
from common import background_filter, common_utils, scan_utils


def get_image_and_info(
    scan_path,
    out_path,
    target_mpp,
    bg_value,
    recompute_image,
    use_cache,
    cache_dir,
):
    if recompute_image:
        if use_cache:
            cache_path = cache_dir.joinpath(scan_path.name)
            common_utils.copy_with_rsync(scan_path, cache_path)
            if scan_path.suffix == ".mrxs":
                # In this case, also copy the corresponding scan folder with the
                # same name and that are placed in the same folder as the scan
                scan_stem = scan_path.stem
                source_scan_folder = scan_path.parent.joinpath(scan_stem)
                assert source_scan_folder.exists()
                common_utils.copy_with_rsync(source_scan_folder, cache_path.parent)
            scan_path = cache_path
        else:
            cache_path = None
        image, info = scan_utils.image_from_scan(scan_path, target_mpp, bg_value)
        if use_cache:
            os.unlink(cache_path)
            if scan_path.suffix == ".mrxs":
                mrxs_folder = cache_path.parent.joinpath(scan_stem)
                assert mrxs_folder.is_dir()
                shutil.rmtree(mrxs_folder)
    else:
        assert out_path.exists(), f"{out_path}"
        image = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
        json_path = out_path.with_suffix(".json")
        assert json_path.exists(), f"{json_path}"
        with json_path.open() as ifile:
            info = json.load(ifile)
    return image, info


def process_scan(
    scan_path,
    out_path,
    target_mpp,
    bg_value,
    mask_classes,
    crop,
    use_cache,
    cache_dir,
    overwrite,
):
    image = None
    info = None
    recompute_image = utils.recompute(out_path, overwrite)
    if mask_classes != ["foreground"]:
        recompute_mask = False
    else:
        recompute_mask = utils.recompute_mask(out_path, mask_classes, overwrite)

    if not (recompute_image or recompute_mask):
        return None

    image, info = get_image_and_info(
        scan_path,
        out_path,
        target_mpp,
        bg_value,
        recompute_image,
        use_cache,
        cache_dir,
    )

    if crop or mask_classes == ["foreground"]:
        foreground_mask = background_filter.background_filter(image) > 0
    else:
        foreground_mask = None

    annotation_mask = None
    if mask_classes == ["foreground"]:
        recompute_mask = True
        mask = background_filter.create_mask(
            mask_classes, annotation_mask, foreground_mask, image
        )
    else:
        recompute_mask = False
        mask = None

    if crop and (recompute_image or recompute_mask):
        image, mask, crop_info = utils.crop_image(image, mask, target_mpp)
        assert crop_info["original_height"] == info["target_height"], "Unequal height"
        assert crop_info["original_width"] == info["target_width"], "Unequal width"
        crop_info.pop("original_height")
        crop_info.pop("original_width")
        info.update(crop_info)

    if recompute_image or recompute_mask:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.with_suffix(".json").open("w") as ofile:
            json.dump(info, ofile)

    if recompute_image:
        cv2.imwrite(str(out_path), image)

    if recompute_mask:
        mask_path = common_utils.get_mask_path(out_path, mask_classes)
        cv2.imwrite(str(mask_path), mask)

    return info


def process_scans(
    scan_paths,
    scan_root,
    out_root,
    target_mpp,
    bg_value,
    add_scan_folder,
    mask_classes,
    crop,
    use_cache,
    cache_dir,
    overwrite,
):
    out_root = scan_root if out_root is None else Path(out_root)
    error_scans = []
    records = []
    for scan_path in tqdm(scan_paths, disable=(len(scan_paths) == 1)):
        out_path = common_utils.output_path_from_scan(
            scan_path, scan_root, out_root
        )
        if add_scan_folder:
            out_path = out_path.parent.joinpath(out_path.stem, out_path.name)
        try:
            json_info = process_scan(
                scan_path,
                out_path,
                target_mpp,
                bg_value,
                mask_classes,
                crop,
                use_cache,
                cache_dir,
                overwrite,
            )
        except Exception as err:
            error_scans.append((scan_path, err))
            log.error(f"Skip due to error in processing: {scan_path}")
            log.error(err)
        else:
            if json_info is not None:
                block = common_utils.block_from_scan_path(scan_path)
                cohort = block.split("-")[0]
                scanner = scan_path.parents[1].name
                record = {
                    "cohort": cohort,
                    "block": block,
                    "scanner": scanner,
                }
                record.update(json_info)
                records.append(record)
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=1)))
    now_formatted = (
        f"{now.year:04}-{now.month:02}-{now.day:02}_"
        f"{now.hour:02}-{now.minute:02}-{now.second:02}"
    )
    if len(records) > 0:
        info_path = out_root.joinpath(f"info_{now_formatted}.csv")
        pd.DataFrame(records).to_csv(info_path, index=False)
    if len(error_scans) > 0:
        out_root.mkdir(parents=True, exist_ok=True)
        with out_root.joinpath(f"errors_{now_formatted}.csv").open("w") as ofile:
            ofile.write("Path,Reason\n")
            for r in error_scans:
                ofile.write(f'{str(r[0])},"{r[1]}"\n')


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--scans",
        metavar="PATH",
        nargs="+",
        required=True,
        help="List of scan paths or root folder with scans below",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Output root folder. Dry run if not given.",
    )
    parser.add_argument(
        "--mpp",
        metavar="FLOAT",
        type=float,
        required=True,
        help="Target resolution in micrometers per pixel",
    )
    parser.add_argument(
        "--mask",
        nargs="+",
        type=str,
        choices=["foreground", "annotation"],
        help="What masks to include in the output mask image.\n"
        "Pixel values in output mask image:\n"
        "255: Annotated tumour if 'annotation' in --mask.\n"
        "127: Automatic segmented foreground if 'foreground' in --mask.\n"
        "  0: Everything else.",
    )
    parser.add_argument(
        "--overwrite",
        metavar="STR",
        choices=["y", "yes", "n", "no"],
        help="If result file exist, overwrite (y/yes) or skip (n/no)",
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
        "--add_scan_folder",
        action="store_true",
        help=(
            "If input scans are all in the same folder, the output will also be so if "
            " this is not set"
        )
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Crop image and mask along the smallest bounding box covering foreground",
    )
    parser.add_argument(
        "--cache",
        metavar="PATH",
        help=(
            "Copy scans to temporary local cache and load scans from there. "
            "Specify cache path here, otherwise no copying happens."
        ),
    )
    args = parser.parse_args()

    common_utils.setup_logging()
    global log
    log = logging.getLogger()

    target_mpp = float(args.mpp)
    # Default background value is used in empty regions only
    # - if the scan file contain empty regions
    # - and if openslide property PROPERTY_NAME_BACKGROUND_COLOR is not present
    # Alpha channel 0 is used to identify empty regions
    # 204 = 0.8 * 255. (204, 204, 204) = #CCCCCC
    bg_value = "CCCCCC"

    if args.scanners is None:
        scan_extensions = set(common_utils.supported_scanners.values())
    else:
        scan_extensions = common_utils.extensions_from_scanners(args.scanners)
    args.scans = [Path(p) for p in args.scans]
    log.info(f"Looking for scans with extensions {scan_extensions}")
    scan_paths, scan_root = common_utils.find_files(args.scans, scan_extensions)
    if scan_paths is None:
        return
    plural = common_utils.plural_s(len(scan_paths))
    log.info(f"Found {len(scan_paths)} input scan file{plural}")
    overwrite = None if args.overwrite is None else args.overwrite in ["y", "yes"]

    if args.output is None:
        log.info("Output path is not given, terminating program")
        return
    if len(scan_paths) == 0:
        log.error("No input scans to resize")
        return
    plural = common_utils.plural_s(len(scan_paths))
    log.info(f"Processing {len(scan_paths)} input scan file{plural}")
    if args.cache is None:
        use_cache = False
        cache_root = Path("/tmp")  # Will not be used, but cannot be None in tempfile
    else:
        use_cache = True
        cache_root = Path(args.cache)
        cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(cache_root)) as tmp_cache_dir:
        process_scans(
            scan_paths,
            scan_root,
            args.output,
            target_mpp,
            bg_value,
            args.add_scan_folder,
            args.mask,
            args.crop,
            use_cache,
            Path(tmp_cache_dir),
            overwrite,
        )


if __name__ == "__main__":
    main()
