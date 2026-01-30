use crate::utils;
use ndarray::{s, Array2, ArrayView, Ix2};
use std::collections::HashMap;

fn mean<T>(elements: T) -> Option<f32>
where
    T: ExactSizeIterator<Item = f32>,
{
    let len = elements.len();
    if len == 0 {
        None
    } else {
        Some(elements.sum::<f32>() / len as f32)
    }
}

fn reduce_diff_standard(a1: &ArrayView<u8, Ix2>, a2: &ArrayView<u8, Ix2>, absolute: bool) -> f32 {
    let v1 = a1.iter();
    let mean1 = if v1.len() == 0 {
        0.0
    } else {
        mean(v1.map(|&x| x as f32 / 255.0)).unwrap()
    };
    let v2 = a2.iter();
    let mean2 = if v2.len() == 0 {
        0.0
    } else {
        mean(v2.map(|&x| x as f32 / 255.0)).unwrap()
    };
    if absolute {
        (mean1 - mean2).abs()
    } else {
        mean1 - mean2
    }
}

/// For each tile, check the average difference in the overlapping region with adjacent tiles.
/// If the max difference is above some threshold, shift the value of this tile with an amount
/// equal to this max difference
fn align_once(
    tiles: &HashMap<utils::Range2D, utils::Tile>,
    threshold: f32,
    shift_down: bool,
) -> (HashMap<utils::Range2D, utils::Tile>, bool) {
    let mut modified_tiles = HashMap::<utils::Range2D, utils::Tile>::new();
    for (r, t) in tiles {
        modified_tiles.insert(r.clone(), t.clone());
    }
    let mut changed = false;

    // Walk through tiles sorted. Easier for debug.
    let mut this_ranges: Vec<utils::Range2D> = tiles.iter().map(|(r, _)| r).cloned().collect();
    this_ranges.sort();

    let mut max_max_diff = Vec::<f32>::new();
    for this_range2d in this_ranges.iter() {
        let this_tile = modified_tiles.get(this_range2d).unwrap();
        assert_eq!(this_range2d, &this_tile.range2d());
        let this_image = this_tile.image().clone();
        let mut overlap_diff = Vec::<f32>::new();
        let mut internal_diff = Vec::<f32>::new();

        // Gather difference between overlapping tiles directly above, below, to the left, and to
        // the rights.

        // Adjacent tile directly above
        if let Some(other_range2d) = this_tile.max_overlapping_tile_direct_top() {
            let d = utils::overlap_top(this_range2d, &other_range2d)
                .unwrap()
                .len() as isize;
            let other_image = modified_tiles.get(&other_range2d).unwrap().image().clone();
            let this_part = this_image.slice(s![..d, ..]);
            let rest_part = this_image.slice(s![d.., ..]);
            let other_part = other_image.slice(s![-d.., ..]);
            assert_eq!(this_part.dim(), other_part.dim());
            overlap_diff.push(reduce_diff_standard(&this_part, &other_part, false));
            internal_diff.push(reduce_diff_standard(&this_part, &rest_part, true));
        }

        // Adjacent tile directly below
        if let Some(other_range2d) = this_tile.max_overlapping_tile_direct_bottom() {
            let d = utils::overlap_bottom(this_range2d, &other_range2d)
                .unwrap()
                .len() as isize;
            let other_image = modified_tiles.get(&other_range2d).unwrap().image().clone();
            let this_part = this_image.slice(s![-d.., ..]);
            let rest_part = this_image.slice(s![..-d, ..]);
            let other_part = other_image.slice(s![..d, ..]);
            assert_eq!(this_part.dim(), other_part.dim());
            overlap_diff.push(reduce_diff_standard(&this_part, &other_part, false));
            internal_diff.push(reduce_diff_standard(&this_part, &rest_part, true));
        }

        // Adjacent tile directly to the left
        if let Some(other_range2d) = this_tile.max_overlapping_tile_direct_left() {
            let d = utils::overlap_left(this_range2d, &other_range2d)
                .unwrap()
                .len() as isize;
            let other_image = modified_tiles.get(&other_range2d).unwrap().image().clone();
            let this_part = this_image.slice(s![.., ..d]);
            let rest_part = this_image.slice(s![.., d..]);
            let other_part = other_image.slice(s![.., -d..]);
            assert_eq!(this_part.dim(), other_part.dim());
            overlap_diff.push(reduce_diff_standard(&this_part, &other_part, false));
            internal_diff.push(reduce_diff_standard(&this_part, &rest_part, true));
        }

        // Adjacent tile directly to the right
        if let Some(other_range2d) = this_tile.max_overlapping_tile_direct_right() {
            let d = utils::overlap_right(this_range2d, &other_range2d)
                .unwrap()
                .len() as isize;
            let other_image = modified_tiles.get(&other_range2d).unwrap().image().clone();
            let this_part = this_image.slice(s![.., -d..]);
            let rest_part = this_image.slice(s![.., ..-d]);
            let other_part = other_image.slice(s![.., ..d]);
            assert_eq!(this_part.dim(), other_part.dim());
            overlap_diff.push(reduce_diff_standard(&this_part, &other_part, false));
            internal_diff.push(reduce_diff_standard(&this_part, &rest_part, true));
        }

        // Shift this tile if the difference is too large
        if !overlap_diff.is_empty() {
            let mut max_index = None;
            let mut max_overlap_diff = -1.0;
            if shift_down {
                for (i, &diff) in overlap_diff.iter().enumerate() {
                    if diff > max_overlap_diff && internal_diff[i] < 0.1 {
                        max_index = Some(i);
                        max_overlap_diff = diff;
                    }
                }
            } else {
                for (i, &diff) in overlap_diff.iter().enumerate() {
                    if diff > max_overlap_diff {
                        max_index = Some(i);
                        max_overlap_diff = diff;
                    }
                }
            }
            max_max_diff.push(max_overlap_diff);

            let mut shift_value = None;
            if max_index.is_some() && max_overlap_diff > threshold {
                if shift_down {
                    shift_value = Some(max_overlap_diff);
                } else {
                    shift_value = Some(mean(overlap_diff.iter().copied()).unwrap());
                }
            }

            if let Some(v) = shift_value {
                changed = true;
                let new_image = Array2::from_shape_vec(
                    this_image.dim(),
                    this_image
                        .iter()
                        .map(|&x| (x as i32 - ((v * 255.0).round() as i32)).max(0).min(255) as u8)
                        .collect(),
                )
                .unwrap();
                let new_tile = this_tile.clone_with_image(&new_image);
                modified_tiles.insert(this_range2d.clone(), new_tile);
            }
        }
    }

    let _max_diff = if max_max_diff.is_empty() {
        None
    } else {
        Some(
            max_max_diff
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
        )
    };
    (modified_tiles.clone(), changed)
}

pub fn align(
    tiles: &HashMap<utils::Range2D, utils::Tile>,
    down_threshold: f32,
    centre_threshold: f32,
) -> HashMap<utils::Range2D, utils::Tile> {
    let mut output_tiles = tiles.clone();
    let mut changed: bool;
    for _ in 0..50 {
        // TODO: Foreground mask
        (output_tiles, changed) = align_once(&output_tiles, down_threshold, true);
        if !changed {
            break;
        }
    }
    (output_tiles, _) = align_once(&output_tiles, centre_threshold, false);
    output_tiles
}
