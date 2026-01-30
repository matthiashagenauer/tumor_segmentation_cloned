use crate::utils;
use ndarray::Array2;
use num_traits::Num;

/// How to combine overlapping tiles.
///
/// This is done either by a weighted average, by the min of all tiles, or by the max of all tiles.
///
/// In case of the weighted average, the final image is computed as
///
/// im = w_1 * tile_1 + ... + w_n * tile_n
///
/// where w_i and tile_i are weight and tile images with and extended domain to the whole merged
/// image.
#[derive(Hash, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Merge {
    /// result[i, j] = min({ones} + {tile_k: (i, j) in tile_k})
    Min,
    /// result[i, j] = max({zeros} + {tile_k: (i, j) in tile_k})
    Max,
    /// Compute Min and Max, then make smooth transitions in areas where they differ more than some
    /// value
    MinMax,
    /// Average weighted by uniform weights
    Uniform,
    /// Average weighted by the distance to the non-overlapping region
    Distance,
}

impl Merge {
    pub fn new(id: &str) -> Self {
        match id {
            "min" => Merge::Min,
            "max" => Merge::Max,
            "minmax" => Merge::MinMax,
            "uniform" => Merge::Uniform,
            "distance" => Merge::Distance,
            _ => unreachable!(),
        }
    }
}

pub fn _ranges_min_row(ranges: &[utils::Range2D]) -> Option<usize> {
    ranges.iter().map(|r| r.vertical().start).min()
}

pub fn ranges_max_row(ranges: &[utils::Range2D]) -> Option<usize> {
    ranges.iter().map(|r| r.vertical().end).max()
}

pub fn _ranges_min_col(ranges: &[utils::Range2D]) -> Option<usize> {
    ranges.iter().map(|r| r.horisontal().start).min()
}

pub fn ranges_max_col(ranges: &[utils::Range2D]) -> Option<usize> {
    ranges.iter().map(|r| r.horisontal().end).max()
}

pub fn global_dim_from_ranges(ranges: &[utils::Range2D]) -> Option<(usize, usize)> {
    let height = ranges_max_row(ranges);
    let width = ranges_max_col(ranges);
    match (height, width) {
        (Some(h), Some(v)) => Some((h, v)),
        _ => None,
    }
}

fn _local_dim_from_ranges(ranges: &[utils::Range2D]) -> Option<(usize, usize)> {
    let mut height = None;
    let mut width = None;
    let mut equal = true;
    for range in ranges {
        if height.is_none() {
            height = Some(range.height())
        } else if height != Some(range.height()) {
            equal = false;
        }
        if width.is_none() {
            width = Some(range.width())
        } else if width != Some(range.width()) {
            equal = false;
        }
    }
    match (height, width) {
        (Some(h), Some(v)) => {
            if equal {
                Some((h, v))
            } else {
                None
            }
        }
        _ => None,
    }
}

pub fn initial_image<T: Clone + Num + ndarray::ScalarOperand>(
    ranges: &[utils::Range2D],
    default_value: T,
) -> Array2<T> {
    let (height, width) =
        global_dim_from_ranges(ranges).expect("ERROR: Could not extract dim from ranges");
    Array2::<T>::ones((height, width)) * default_value
}
