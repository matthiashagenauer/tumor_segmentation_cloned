import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from tile import Tile

log = logging.getLogger("align")


def reduce_diff_mean(part_1: np.ndarray, part_2: np.ndarray, absolute: bool = False):
    mean_1 = float(np.mean(part_1)) if len(part_1) > 0 else 0.0
    mean_2 = float(np.mean(part_2)) if len(part_2) > 0 else 0.0
    if absolute:
        return np.abs(mean_1 - mean_2)
    else:
        return mean_1 - mean_2


def align_tiles_once(
    tiles: Dict[Path, Tile],
    foreground_mask: np.ndarray,
    threshold: float,
    shift_down: bool = True,
) -> Tuple[Dict[Path, Tile], bool]:
    max_part_diff = []
    keys = sorted(tiles.keys())
    changed = False
    for this_path in keys:
        this_tile = tiles[this_path]
        this_range2d = this_tile.range2d
        this_image = this_tile.image
        this_mask = foreground_mask[
            this_range2d.top : this_range2d.bottom,
            this_range2d.left : this_range2d.right,
        ]
        overlap_diff = []
        internal_diff = []
        for other_path in keys:
            if other_path == this_path:
                continue
            other_tile = tiles[other_path]
            other_image = other_tile.image
            assert this_image.shape == other_image.shape
            overlap_top = this_tile.direct_overlap_top(other_tile)
            overlap_bottom = this_tile.direct_overlap_bottom(other_tile)
            overlap_left = this_tile.direct_overlap_left(other_tile)
            overlap_right = this_tile.direct_overlap_right(other_tile)
            if overlap_top is not None:
                d = len(overlap_top)
                this_mask_part = this_mask[:d, :]
                rest_mask_part = this_mask[d:, :]
                if np.any(this_mask_part):
                    this_part = this_image[:d, :][this_mask_part]
                    rest_part = this_image[d:, :][rest_mask_part]
                    other_part = other_image[-d:, :][this_mask_part]
                    assert this_part.shape == other_part.shape
                    overlap_diff.append(reduce_diff_mean(this_part, other_part))
                    internal_diff.append(reduce_diff_mean(this_part, rest_part, True))
            if overlap_bottom is not None:
                d = len(overlap_bottom)
                this_mask_part = this_mask[-d:, :]
                rest_mask_part = this_mask[:-d, :]
                if np.any(this_mask_part):
                    this_part = this_image[-d:, :][this_mask_part]
                    rest_part = this_image[:-d, :][rest_mask_part]
                    other_part = other_image[:d, :][this_mask_part]
                    assert this_part.shape == other_part.shape
                    overlap_diff.append(reduce_diff_mean(this_part, other_part))
                    internal_diff.append(reduce_diff_mean(this_part, rest_part, True))
            if overlap_left is not None:
                d = len(overlap_left)
                this_mask_part = this_mask[:, :d]
                rest_mask_part = this_mask[:, d:]
                if np.any(this_mask_part):
                    this_part = this_image[:, :d][this_mask_part]
                    rest_part = this_image[:, d:][rest_mask_part]
                    other_part = other_image[:, -d:][this_mask_part]
                    assert this_part.shape == other_part.shape
                    overlap_diff.append(reduce_diff_mean(this_part, other_part))
                    internal_diff.append(reduce_diff_mean(this_part, rest_part, True))
            if overlap_right is not None:
                d = len(overlap_right)
                this_mask_part = this_mask[:, -d:]
                rest_mask_part = this_mask[:, :-d]
                if np.any(this_mask_part):
                    this_part = this_image[:, -d:][this_mask_part]
                    rest_part = this_image[:, :-d][rest_mask_part]
                    other_part = other_image[:, :d][this_mask_part]
                    assert this_part.shape == other_part.shape
                    overlap_diff.append(reduce_diff_mean(this_part, other_part))
                    internal_diff.append(reduce_diff_mean(this_part, rest_part, True))
        if len(overlap_diff) > 0:
            max_part_diff.append(max([abs(v) for v in overlap_diff]))
            max_index = None
            max_diff = -1.0
            if shift_down:
                for i, diff in enumerate(overlap_diff):
                    if diff > max_diff and internal_diff[i] < 25.0:
                        max_index = i
                        max_diff = diff
            else:
                for i, diff in enumerate(overlap_diff):
                    if diff > max_diff and internal_diff[i] < 25.0:
                        max_index = i
                        max_diff = diff

            shift_value = None
            if max_index is not None:
                if max_diff > threshold:
                    if shift_down:
                        shift_value = max_diff
                    else:
                        shift_value = np.mean(overlap_diff)
                else:
                    shift_value = None
            else:
                shift_value = None

            if shift_value is not None:
                changed = True
                tiles[this_path] = Tile(
                    this_range2d,
                    np.floor(
                        np.clip(
                            this_image.astype(np.float32) - shift_value,
                            0.0,
                            255.0,
                        )
                    ).astype(np.uint8),
                    this_tile.overlap_info,
                )

    max_max_part_diff = None if len(max_part_diff) == 0 else max(max_part_diff)
    log.debug(f"Max mean difference: {max_max_part_diff}")
    return tiles, changed


def align_tiles(
    tiles: Dict[Path, Tile],
    foreground_mask: np.ndarray,
    down_threshold: float = 0.0,
    centre_threshold: float = 0.0,
) -> Dict[Path, Tile]:
    for i in range(50):
        log.debug(f"Align iteration: {i+1:>2}")
        tiles, changed = align_tiles_once(tiles, foreground_mask, down_threshold)
        if not changed:
            break
    tiles, _ = align_tiles_once(tiles, foreground_mask, centre_threshold, False)
    return tiles


def main():
    import sys
    from pathlib import Path
    import time
    import cv2  # type: ignore
    from tqdm import tqdm  # type: ignore
    from merge_predictions import merge_tile_predictions
    from tile import construct_tiles

    sys.path.insert(0, str(Path(__file__).resolve().parents[2].joinpath("postprocess")))
    import segment_probability_maps  # type: ignore

    assert len(sys.argv) == 3
    mask_root_dir = Path(sys.argv[1])
    input_dir = Path(sys.argv[2])
    section = input_dir.name
    input_dir = input_dir.joinpath(section)
    output_dir = Path("/tmp/align_tiles/tilesize-7680")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = list(input_dir.glob("*png"))
    foreground_mask_dir = mask_root_dir.joinpath(section)
    candidates = list(foreground_mask_dir.glob("*foreground*.png"))
    if len(candidates) == 0:
        print("No foreground mask image in", foreground_mask_dir)
        return
    foreground_mask_path = candidates[0]
    predictions = []
    foreground_mask = cv2.imread(str(foreground_mask_path), cv2.IMREAD_GRAYSCALE) > 0

    resize_factor = 0.2
    down_threshold = 85.0
    centre_threshold = 85.0

    start_time = time.time()
    for p in tqdm(input_paths):
        predictions.append(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
    print(f"Read input:        {time.time() - start_time:>5.2f} seconds")

    tiles = construct_tiles(input_paths, predictions, resize_factor)

    original_image = merge_tile_predictions(tiles)
    cv2.imwrite(
        str(output_dir.joinpath(section + "_original.png")),
        original_image,
    )

    start_time = time.time()
    tiles = align_tiles(tiles, foreground_mask, down_threshold, centre_threshold)
    print(f"Align tiles:       {time.time() - start_time:>5.2f} seconds")

    start_time = time.time()
    # weight_output = output_dir.joinpath(section + "_weight_tiles")
    weight_output = None
    aligned_image = merge_tile_predictions(tiles, weight_output)
    print(f"Merge predictions: {time.time() - start_time:>5.2f} seconds")

    cv2.imwrite(
        str(output_dir.joinpath(section + "_realigned.png")),
        aligned_image,
    )

    start_time = time.time()
    mask = segment_probability_maps.segment_image(
        aligned_image,
        None,
        foreground_mask,
        True,
        "dist-204-400-85",
        "percentile-95-229",
        None,
        False,
    )

    class_labels = [0, 255]
    labels = np.unique(mask)
    assert np.max(labels) < len(class_labels), "Incompatible label and classes"
    for label in sorted(labels):
        mask[mask == label] = class_labels[label]
    print(f"Segmentation:      {time.time() - start_time:>5.2f} seconds")
    cv2.imwrite(str(output_dir.joinpath(section + "_segmented.png")), mask)

    print(f"Top 99 percentile: {np.percentile(aligned_image, 99)}")


if __name__ == "__main__":
    main()
