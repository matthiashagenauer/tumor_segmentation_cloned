from collections import namedtuple
import logging
from pathlib import Path
import sys
from typing import Dict, Mapping, Optional, Tuple

import cv2  # type: ignore
import numpy as np

from tile import Range2D, Tile, global_dim_from_ranges

log = logging.getLogger("merge")


def base_array(length: int, overlap: int, reverse=False) -> np.ndarray:
    """
    returns array on the form
    [(1 + i) / (1 + overlap) if i < overlap else 1.0 for i in range(length)]
    """
    assert length > overlap
    array = np.ones(length, dtype=np.float32)
    array[:overlap] = np.linspace(1 / (1 + overlap), overlap / (1 + overlap), overlap)
    if reverse:
        array = array[::-1]
    return array


def side_weight_tile(dim: Tuple[int, int], overlap: int, side: str) -> np.ndarray:
    height, width = dim
    if side == "top":
        weight_tile = np.array([base_array(height, overlap, False)] * width).transpose()
    elif side == "bottom":
        weight_tile = np.array([base_array(height, overlap, True)] * width).transpose()
    elif side == "left":
        weight_tile = np.array([base_array(width, overlap, False)] * height)
    elif side == "right":
        weight_tile = np.array([base_array(width, overlap, True)] * height)
    else:
        log.error(f"Invalid side: {side}")
        sys.exit()
    return weight_tile


def compute_weight_tile(
    range2d: Range2D,
    overlap_top_size: Optional[int],
    overlap_bottom_size: Optional[int],
    overlap_left_size: Optional[int],
    overlap_right_size: Optional[int],
) -> np.ndarray:
    product_weight_tile = np.ones(range2d.dim, dtype=np.float32)
    if overlap_top_size is not None:
        weight_tile = side_weight_tile(range2d.dim, overlap_top_size, "top")
        product_weight_tile *= weight_tile
    if overlap_bottom_size is not None:
        weight_tile = side_weight_tile(range2d.dim, overlap_bottom_size, "bottom")
        product_weight_tile *= weight_tile
    if overlap_left_size is not None:
        weight_tile = side_weight_tile(range2d.dim, overlap_left_size, "left")
        product_weight_tile *= weight_tile
    if overlap_right_size is not None:
        weight_tile = side_weight_tile(range2d.dim, overlap_right_size, "right")
        product_weight_tile *= weight_tile
    return product_weight_tile


WeightTileKey = namedtuple("WeightTileKey", ["size", "top", "bottom", "left", "right"])


def compute_weight_tiles(
    tiles: Mapping[Path, Tile],
    global_dim: Tuple[int, int],
) -> Dict[Path, np.ndarray]:
    weight_tiles: Dict[Path, np.ndarray] = {}
    sum_weight_image = np.zeros(global_dim, dtype=np.float32)
    weight_tile_cache: Dict[WeightTileKey, np.ndarray] = {}
    paths = sorted(list(tiles.keys()))
    for path in paths:
        tile = tiles[path]
        range2d = tile.range2d
        key = WeightTileKey(
            range2d.dim,
            tile.max_overlap_top_size(),
            tile.max_overlap_bottom_size(),
            tile.max_overlap_left_size(),
            tile.max_overlap_right_size(),
        )
        if key in weight_tile_cache:
            weight_tile = weight_tile_cache[key]
        else:
            weight_tile = compute_weight_tile(
                range2d,
                tile.max_overlap_top_size(),
                tile.max_overlap_bottom_size(),
                tile.max_overlap_left_size(),
                tile.max_overlap_right_size(),
            )
            weight_tile_cache[key] = weight_tile
        sum_weight_image[
            range2d.top : range2d.bottom, range2d.left : range2d.right
        ] += weight_tile
        weight_tiles[path] = weight_tile

    for (path, image) in weight_tiles.items():
        range2d = tiles[path].range2d
        weight_tiles[path] = (
            image
            / sum_weight_image[
                range2d.top : range2d.bottom, range2d.left : range2d.right
            ]
        )

    return weight_tiles


def merge_tile_predictions(
    tiles: Dict[Path, Tile],
    weight_tile_output: Optional[Path] = None,
) -> np.ndarray:
    global_dim = global_dim_from_ranges([t.range2d for t in tiles.values()])
    weight_tiles = compute_weight_tiles(tiles, global_dim)

    if weight_tile_output is not None:
        weight_tile_output.mkdir(parents=True, exist_ok=True)
        for path, tile in tiles.items():
            weight_tile = weight_tiles[path]
            cv2.imwrite(
                str(weight_tile_output.joinpath(path.stem)) + ".png",
                np.floor(weight_tile * 255.0).astype(np.uint8),
            )

    result_image = np.zeros(global_dim, dtype=np.uint8)
    for (path, tile) in tiles.items():
        weighted_image = np.floor(
            np.clip(
                tile.image.astype(np.float32) * weight_tiles[path],
                0.0,
                255.0,
            )
        ).astype(np.uint8)
        result_image[
            tile.range2d.top : tile.range2d.bottom,
            tile.range2d.left : tile.range2d.right,
        ] += weighted_image

    return result_image


def main():
    import sys
    from tqdm import tqdm  # type: ignore
    from tile import construct_tiles

    input_dir = Path(sys.argv[1])
    output_dir = Path("/tmp/merged_tile_predictions")
    output_dir.mkdir(exist_ok=True)
    input_paths = list(input_dir.glob("*.png"))
    tile_predictions = []
    for p in tqdm(input_paths):
        tile_predictions.append(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
    tiles = construct_tiles(input_paths, tile_predictions, None, None)
    image = merge_tile_predictions(tiles, output_dir.joinpath("weights"))
    cv2.imwrite(str(output_dir.joinpath(input_dir.name + ".png")), image)


if __name__ == "__main__":
    main()
