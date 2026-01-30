"""
This script unites all postprocessing of the probability maps produced by the
segmentation network. The main operations are, in this order:

- Segment probability maps using conv CRF [ref1], based on dense CRF [ref2]
- Remove small objects and apply a simple background filter
- Write annotation file in the original dimension

Ole-Johan Skrede
26.11.2020


References:

[1] Convolutional CRFs for Semantic Segmentation
    Marvin T. T. Teichmann, Roberto Cipolla
    Arxiv preprint, https://arxiv.org/abs/1805.04777
    https://github.com/MarvinTeichmann/ConvCRF

[2] Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
    Philipp Krähenbühl and Vladlen Koltun
    NIPS 2011
    https://github.com/lucasb-eyer/pydensecrf.git
"""

import argparse
from pathlib import Path
import time
import logging

import cv2
import numpy as np
from scipy import ndimage
import json

from common import background_filter, common_utils
from prepare_mask import prepare_mask
import utils


def get_foreground_mask(reference_path, image_bgr):
    if reference_path is None:
        if image_bgr is None:
            foreground_mask = np.ones((image_bgr.shape[0], image_bgr.shape[1])) > 0
        else:
            foreground_mask = background_filter.background_filter(image_bgr=image_bgr)
    else:
        reference_mask = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        foreground_mask = reference_mask > 0
    return foreground_mask


def segment_image(
    probability_map,
    image_bgr,
    foreground_mask,
    smooth,
    method,
    prune,
    gpu,
    close_holes,
):
    height, width = probability_map.shape
    mask, probability_map = prepare_mask(
        probability_map, image_bgr, smooth, method, prune, gpu
    )
    mask = mask & foreground_mask
    if close_holes:
        mask = ndimage.binary_fill_holes(mask.astype(np.uint8)).astype(bool)
    mask = background_filter.background_filter(input_mask=mask.astype(np.uint8))
    return mask, probability_map


def process_single(paths, smooth, method, prune, gpu, close_holes):
    assert paths.input.exists(), f"Input path does not exist: {paths.input}"
    if paths.image is not None:
        assert paths.image.exists(), f"Image path does not exist: {paths.image}"
    class_labels = [0, 255]

    probability_map = cv2.imread(str(paths.input), cv2.IMREAD_GRAYSCALE)
    if (paths.reference is None and paths.image is not None) or method == "crf":
        image_bgr = cv2.imread(str(paths.image), cv2.IMREAD_COLOR)
    else:
        image_bgr = None

    foreground_mask = get_foreground_mask(paths.reference, image_bgr)
    mask, probability_map = segment_image(
        probability_map,
        image_bgr,
        foreground_mask,
        smooth,
        method,
        prune,
        gpu,
        close_holes,
    )
    mask = utils.class_index_to_label(mask, class_labels)

    if paths.mask is not None:
        paths.mask.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(paths.mask), mask)
    if paths.probability is not None:
        paths.probability.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(paths.probability), probability_map)


def process_multiple(
    input_paths,
    input_root,
    image_root,
    mask_root,
    prob_root,
    smooth,
    method,
    prune,
    gpu,
    close_holes,
    overwrite,
):
    logging.info(
        "{:>18}{:>30}{:>10}{:>10}".format(
            "Progress",
            "Name",
            "Elapsed",
            "Average",
        )
    )
    num_inputs = len(input_paths)
    sum_elapsed = 0
    num_complete = 0
    for i, input_path in enumerate(sorted(input_paths)):
        start_time = time.time()
        paths = utils.find_corresponding(
            input_path,
            input_root,
            image_root,
            mask_root,
            prob_root,
            overwrite,
        )
        if paths is not None:
            try:
                process_single(paths, smooth, method, prune, gpu, close_holes)
            except Exception as err:
                logging.error(f"Error processing {input_path}")
                logging.error(err)
            else:
                elapsed = time.time() - start_time
                name = input_path.name

                sum_elapsed += elapsed
                num_complete += 1
                percent_complete = 100 * (i + 1) / num_inputs

                progress = f"{i + 1:>4} / {num_inputs:>4} = {percent_complete:>3.0f}%"

                logging.info(
                    "{}{:>30}{:>10.2f}{:>10.2f}".format(
                        progress,
                        name,
                        elapsed,
                        sum_elapsed / num_complete,
                    )
                )


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "input",
        metavar="PATH",
        nargs="+",
        help="List of probability map input paths",
    )
    parser.add_argument(
        "--images",
        metavar="PATH",
        help=(
            "Folder with images (.png) and resize info (.json) corresponding to \n"
            "input. It is assumed that images and resize info share the same path \n"
            "except for their extension. It is also assumed that the path of images \n"
            "and resize info file relative to this input folder is the same as the \n"
            "path of their corresponding mask relative to the common input folder of \n"
            "all input masks"
        ),
    )
    parser.add_argument(
        "--output_mask",
        metavar="PATH",
        required=True,
        help=(
            "Output root folder for segmentation image files. If this is given, png \n"
            "files are written at the downscaled dimension"
        ),
    )
    parser.add_argument(
        "--output_prob",
        metavar="PATH",
        help=(
            "Output root folder for segmentation image files. If this is given, png \n"
            "files are written at the downscaled dimension"
        ),
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Smooth probability maps before generating masks",
    )
    parser.add_argument(
        "--method",
        metavar="STR",
        default="argmax",
        help=(
            "Decide how to generate masks from probability maps. Default 'argmax'.\n"
            "Choices:\n"
            "   argmax\n"
            "   threshold-<value in [0,255)>\n"
            "   reconstruction-<value in [0,255)>-<value in [0,255)>\n"
            "   crf\n"
        ),
    )
    parser.add_argument(
        "--prune",
        metavar="STR",
        help=(
            "Remove mask foreground regions not fulfilling some criteria. Input is \n"
            "on the form 'percentile-<value in (0, 100)>-<value in [0, 255)>."
        ),
    )
    parser.add_argument(
        "--gpu",
        metavar="INT",
        type=int,
        default=0,
        help="Specify which gpu device to run convolutional crf on. Default 0",
    )
    parser.add_argument(
        "--overwrite",
        metavar="STR",
        choices=["y", "yes", "n", "no"],
        help="Decide what to do if results exist: y / yes or n / no",
    )
    parser.add_argument(
        "--close",
        action="store_true",
        help="Close holes before applying background filtering",
    )
    args = parser.parse_args()
    args.input = [Path(p) for p in args.input]

    image_root = None if args.images is None else Path(args.images)

    mask_root = Path(args.output_mask)
    prob_root = None if args.output_prob is None else Path(args.output_prob)
    if mask_root is None:
        logging.error("--output_mask must be given")
        return
    log_dir = None
    if mask_root is not None:
        log_dir = mask_root
    if log_dir is not None and log_dir.suffix != "":
        log_dir = log_dir.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = None if log_dir is None else log_dir.joinpath("postprocess.log")
    common_utils.setup_logging(logfile=logfile)
    if args.method == "crf" and image_root is None:
        logging.error("CRF requires input images")
        return
    mask_extensions = [".png"]
    input_paths, input_root = common_utils.find_files(args.input, mask_extensions)
    logging.info(f"Found {len(input_paths)} input probability maps")

    start_time = time.time()
    if len(input_paths) == 0:
        logging.error("No probability maps to segment")
        return
    elif len(input_paths) == 1:
        paths = utils.find_corresponding(
            input_paths[0],
            input_root,
            image_root,
            mask_root,
            prob_root,
            args.overwrite,
        )
        if paths is not None:
            process_single(
                paths, args.smooth, args.method, args.prune, args.gpu, args.close
            )
    else:
        process_multiple(
            input_paths,
            input_root,
            image_root,
            mask_root,
            prob_root,
            args.smooth,
            args.method,
            args.prune,
            args.gpu,
            args.close,
            args.overwrite,
        )
    logging.info(f"Total elapsed: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
