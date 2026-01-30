use crate::restore::restore_utils;
use crate::utils;
use ndarray::{s, Array, Array2};
use std::{collections::HashMap, ops::AddAssign, path::PathBuf};

enum Side {
    Top,
    Bottom,
    Left,
    Right,
}

fn uniform_overlap_array(length: usize, overlap: usize) -> Vec<f32> {
    let mut array: Vec<f32> = vec![1.0; length];
    // for i in 0..overlap {
    for elem in array.iter_mut().take(overlap) {
        *elem = 0.5;
    }
    array
}

fn distance_overlap_array(length: usize, overlap: usize) -> Vec<f32> {
    let mut array: Vec<f32> = vec![1.0; length];
    for (x, elem) in array.iter_mut().enumerate().take(overlap) {
        *elem = (1.0 + x as f32) / (1.0 + overlap as f32)
    }
    array
}

fn base_array(length: usize, overlap: usize, strategy: restore_utils::Merge) -> Vec<f32> {
    match strategy {
        restore_utils::Merge::Uniform => uniform_overlap_array(length, overlap),
        restore_utils::Merge::Distance => distance_overlap_array(length, overlap),
        _ => unreachable!(),
    }
}

fn side_weight_tile(
    shape: [usize; 2],
    overlap: usize,
    strategy: restore_utils::Merge,
    side: Side,
) -> Array2<f32> {
    let mut weight_tile = Array2::<f32>::ones(shape);
    let [height, width] = shape;
    match side {
        Side::Top => {
            let array = base_array(height, overlap, strategy);
            for mut column in weight_tile.columns_mut() {
                column.assign(&Array::from_vec(array.clone()));
            }
        }
        Side::Bottom => {
            let mut array = base_array(height, overlap, strategy);
            array.reverse();
            for mut column in weight_tile.columns_mut() {
                column.assign(&Array::from_vec(array.clone()));
            }
        }
        Side::Left => {
            let array = base_array(width, overlap, strategy);
            for mut row in weight_tile.rows_mut() {
                row.assign(&Array::from_vec(array.clone()));
            }
        }
        Side::Right => {
            let mut array = base_array(width, overlap, strategy);
            array.reverse();
            for mut row in weight_tile.rows_mut() {
                row.assign(&Array::from_vec(array.clone()));
            }
        }
    }
    weight_tile
}

#[derive(Hash, Debug, PartialEq, Eq)]
pub struct WeightTileKey {
    strategy: restore_utils::Merge,
    size: [usize; 2],
    overlap_top: Option<usize>,
    overlap_bottom: Option<usize>,
    overlap_left: Option<usize>,
    overlap_right: Option<usize>,
}

pub fn compute_weight_tile(
    range2d: &utils::Range2D,
    overlap_top_size: Option<usize>,
    overlap_bottom_size: Option<usize>,
    overlap_left_size: Option<usize>,
    overlap_right_size: Option<usize>,
    strategy: restore_utils::Merge,
) -> Array2<f32> {
    let height = range2d.height();
    let width = range2d.width();
    let mut product_weight_tile = Array2::<f32>::ones((height, width));
    if let Some(size) = overlap_top_size {
        let top_weight_tile = side_weight_tile([height, width], size, strategy, Side::Top);
        product_weight_tile = product_weight_tile * top_weight_tile;
    }
    if let Some(size) = overlap_bottom_size {
        let bottom_weight_tile = side_weight_tile([height, width], size, strategy, Side::Bottom);
        product_weight_tile = product_weight_tile * bottom_weight_tile;
    }
    if let Some(size) = overlap_left_size {
        let left_weight_tile = side_weight_tile([height, width], size, strategy, Side::Left);
        product_weight_tile = product_weight_tile * left_weight_tile;
    }
    if let Some(size) = overlap_right_size {
        let right_weight_tile = side_weight_tile([height, width], size, strategy, Side::Right);
        product_weight_tile = product_weight_tile * right_weight_tile;
    }
    product_weight_tile
}

/// Compute weights to be multiplied with the image content tile before they are added to the final
/// merged image. The weight tiles are computed in 3 steps
///
/// 1: Initial weights are computed. Find the max overlap size in each spatial direction (up, down,
///    left, right), and change the values in these regions based on merge strategy.
///
/// 2: Add all initial weight tiles to a global weight image with 'global_dim' shape
///
/// 3: Normalise the initial weight tiles by dividing them with the content of the global sum image
///    in their location.
///
fn compute_weight_tiles(
    tiles: &HashMap<utils::Range2D, utils::Tile>,
    global_dim: (usize, usize),
    strategy: restore_utils::Merge,
) -> HashMap<utils::Range2D, Array2<f32>> {
    let mut weight_tiles = HashMap::<utils::Range2D, Array2<f32>>::new();
    let mut sum_weight_image = Array2::<f32>::zeros(global_dim);
    // NOTE: Caching does not make a large difference when there are few tiles. Most time seems to
    // be spent on reading tiles anyway
    let mut weight_tile_cache = HashMap::<WeightTileKey, Array2<f32>>::new();
    for (range2d, tile) in tiles.iter() {
        let key = WeightTileKey {
            strategy,
            size: [range2d.height(), range2d.width()],
            overlap_top: tile.max_overlap_top().map(|r| r.len()),
            overlap_bottom: tile.max_overlap_bottom().map(|r| r.len()),
            overlap_left: tile.max_overlap_left().map(|r| r.len()),
            overlap_right: tile.max_overlap_right().map(|r| r.len()),
        };
        let weight_tile = weight_tile_cache
            .entry(key)
            .or_insert_with(|| {
                compute_weight_tile(
                    range2d,
                    tile.max_overlap_top().map(|r| r.len()),
                    tile.max_overlap_bottom().map(|r| r.len()),
                    tile.max_overlap_left().map(|r| r.len()),
                    tile.max_overlap_right().map(|r| r.len()),
                    strategy,
                )
            })
            .clone();
        sum_weight_image
            .slice_mut(s![range2d.vertical(), range2d.horisontal()])
            .add_assign(&weight_tile);
        weight_tiles.insert(range2d.clone(), weight_tile);
    }

    for (range2d, im) in weight_tiles.iter_mut() {
        let local_sum = sum_weight_image.slice(s![range2d.vertical(), range2d.horisontal()]);
        *im = im.clone() / local_sum;
    }

    weight_tiles
}

pub fn merge_tiles_average_exact(
    tiles: &HashMap<utils::Range2D, utils::Tile>,
    debug_dir: Option<PathBuf>,
    strategy: restore_utils::Merge,
) -> Array2<f32> {
    let ranges: Vec<utils::Range2D> = tiles.keys().cloned().collect();
    let mut result_image = restore_utils::initial_image(&ranges, 0.0);
    let mut sum_weight_image = restore_utils::initial_image(&ranges, 0.0);
    let global_dim = result_image.dim();

    if debug_dir.is_some() {
        println!("Write debug to {:?}", debug_dir)
    }

    let weight_tiles = compute_weight_tiles(tiles, global_dim, strategy);

    for (range2d, tile) in tiles {
        let float_tile = tile.image().map(|&v| v as f32);
        let weight_tile = weight_tiles.get(range2d).unwrap();
        let result_tile = float_tile * weight_tile;
        result_image
            .slice_mut(s![range2d.vertical(), range2d.horisontal()])
            .add_assign(&result_tile);
        if let Some(ref dir) = debug_dir {
            sum_weight_image
                .slice_mut(s![range2d.vertical(), range2d.horisontal()])
                .add_assign(weight_tile);
            utils::save_array(
                &dir.join(format!("weight-tile_{}.png", range2d)),
                &weight_tile.map(|v| (v * 255.0).round() as u8),
            )
            .unwrap();
        }
    }
    if debug_dir.is_some() {
        println!(
            "Min weight sum image: {}",
            sum_weight_image
                .iter()
                .fold(f32::INFINITY, |a, &b| a.min(b))
        );
        println!(
            "Max weight sum image: {}",
            sum_weight_image
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        );
    }
    result_image
}

pub fn merge_tiles_average_round(
    tiles: &HashMap<utils::Range2D, utils::Tile>,
    debug_dir: Option<PathBuf>,
    strategy: restore_utils::Merge,
) -> Array2<u8> {
    let ranges: Vec<utils::Range2D> = tiles.keys().cloned().collect();
    let mut result_image = restore_utils::initial_image(&ranges, 0);
    let mut sum_weight_image = restore_utils::initial_image(&ranges, 0.0);
    let global_dim = result_image.dim();

    let weight_tiles = compute_weight_tiles(tiles, global_dim, strategy);

    for (range2d, tile) in tiles {
        let float_tile = tile.image().map(|&v| v as f32);
        let weight_tile = weight_tiles.get(range2d).unwrap();
        let result_tile = (float_tile * weight_tile).map(|v| v.round() as u8);
        result_image
            .slice_mut(s![range2d.vertical(), range2d.horisontal()])
            .add_assign(&result_tile);
        if debug_dir.is_some() {
            sum_weight_image
                .slice_mut(s![range2d.vertical(), range2d.horisontal()])
                .add_assign(weight_tile);
        }
    }
    if debug_dir.is_some() {
        println!(
            "Min weight sum image: {}",
            sum_weight_image
                .iter()
                .fold(f32::INFINITY, |a, &b| a.min(b))
        );
        println!(
            "Max weight sum image: {}",
            sum_weight_image
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        );
    }
    result_image
}

#[cfg(test)]
mod tests {
    use super::{compute_weight_tile, merge_tiles_average_exact, restore_utils::Merge};
    use crate::utils;
    use ndarray::{arr2, Array2};
    use std::collections::HashMap;

    fn small_tile_hor_a() -> (utils::Range2D, Array2<u8>) {
        let tile = arr2(&[[2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2]]);
        let range_2d = utils::Range2D::new(0..2, 0..8);
        (range_2d, tile)
    }

    fn small_tile_hor_b() -> (utils::Range2D, Array2<u8>) {
        let tile = arr2(&[[4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4]]);
        let range_2d = utils::Range2D::new(0..2, 3..11);
        (range_2d, tile)
    }

    fn small_tile_hor_c() -> (utils::Range2D, Array2<u8>) {
        let tile = arr2(&[[8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8]]);
        let range_2d = utils::Range2D::new(0..2, 7..15);
        (range_2d, tile)
    }

    #[test]
    fn test_distance_interpolation_ab_left() {
        let (range_a, _) = small_tile_hor_a();
        let weight_tile =
            compute_weight_tile(&range_a.clone(), None, None, Some(5), None, Merge::Distance);
        let expected_weight_tile = arr2(&[
            [
                1.0 / 6.0,
                2.0 / 6.0,
                3.0 / 6.0,
                4.0 / 6.0,
                5.0 / 6.0,
                1.0,
                1.0,
                1.0,
            ],
            [
                1.0 / 6.0,
                2.0 / 6.0,
                3.0 / 6.0,
                4.0 / 6.0,
                5.0 / 6.0,
                1.0,
                1.0,
                1.0,
            ],
        ]);
        utils::compare_arrays(&weight_tile, &expected_weight_tile, "Distance left");
    }

    #[test]
    fn test_distance_interpolation_ab_right() {
        let (range_a, _) = small_tile_hor_a();
        let weight_tile =
            compute_weight_tile(&range_a.clone(), None, None, None, Some(5), Merge::Distance);
        let expected_weight_tile = arr2(&[
            [
                1.0,
                1.0,
                1.0,
                5.0 / 6.0,
                4.0 / 6.0,
                3.0 / 6.0,
                2.0 / 6.0,
                1.0 / 6.0,
            ],
            [
                1.0,
                1.0,
                1.0,
                5.0 / 6.0,
                4.0 / 6.0,
                3.0 / 6.0,
                2.0 / 6.0,
                1.0 / 6.0,
            ],
        ]);
        utils::compare_arrays(&weight_tile, &expected_weight_tile, "Distance right");
    }

    #[test]
    fn test_distance_interpolation_hor_result_ab() {
        let (range_a, tile_a) = small_tile_hor_a();
        let (range_b, tile_b) = small_tile_hor_b();
        let mut tiles = HashMap::<utils::Range2D, utils::Tile>::new();
        let all_ranges = [range_a.clone(), range_b.clone()];
        tiles.insert(
            range_a.clone(),
            utils::Tile::new(&range_a, &tile_a.clone(), &all_ranges),
        );
        tiles.insert(
            range_b.clone(),
            utils::Tile::new(&range_b, &tile_b.clone(), &all_ranges),
        );
        let result = merge_tiles_average_exact(&tiles, None, Merge::Distance);
        let expected_result = arr2(&[
            [
                2.0,
                2.0,
                2.0,
                7.0 / 3.0,
                8.0 / 3.0,
                3.0,
                10.0 / 3.0,
                11.0 / 3.0,
                4.0,
                4.0,
                4.0,
            ],
            [
                2.0,
                2.0,
                2.0,
                7.0 / 3.0,
                8.0 / 3.0,
                3.0,
                10.0 / 3.0,
                11.0 / 3.0,
                4.0,
                4.0,
                4.0,
            ],
        ]);
        utils::compare_arrays(&result, &expected_result, "Result");
    }

    // NOTE: The below tests assumes that it is allowed for tiles to overlap with more than one
    // tile in a single direction. This is not implemented yes and will throw an error. Example
    //
    // |---------------|-----------|--------------|----------|-------------|
    // Start 1         Start 2     Start 3        End 1      End 2         End 3
    //

    #[test]
    fn test_distance_interpolation_ac_left() {
        let (range_c, _) = small_tile_hor_c();
        let weight_tile =
            compute_weight_tile(&range_c.clone(), None, None, Some(1), None, Merge::Distance);
        let expected_weight_tile = arr2(&[
            [1.0 / 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0 / 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]);
        utils::compare_arrays(
            &weight_tile,
            &expected_weight_tile,
            "Distance weight_tile cb",
        );
    }

    #[test]
    fn test_distance_interpolation_bc_left() {
        let (range_c, _) = small_tile_hor_c();
        let weight_tile =
            compute_weight_tile(&range_c.clone(), None, None, Some(4), None, Merge::Distance);
        let expected_weight_tile = arr2(&[
            [
                1.0 / 5.0,
                2.0 / 5.0,
                3.0 / 5.0,
                4.0 / 5.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            [
                1.0 / 5.0,
                2.0 / 5.0,
                3.0 / 5.0,
                4.0 / 5.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        ]);
        utils::compare_arrays(
            &weight_tile,
            &expected_weight_tile,
            "Distance weight_tile cb",
        );
    }

    #[test]
    fn test_distance_interpolation_ac_right() {
        let (range_a, _) = small_tile_hor_a();
        let weight_tile =
            compute_weight_tile(&range_a.clone(), None, None, None, Some(1), Merge::Distance);
        let expected_weight_tile = arr2(&[
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 / 2.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 / 2.0],
        ]);
        utils::compare_arrays(
            &weight_tile,
            &expected_weight_tile,
            "Distance weight_tile ac",
        );
    }

    #[test]
    fn test_distance_interpolation_bc_right() {
        let (range_b, _) = small_tile_hor_b();
        let weight_tile =
            compute_weight_tile(&range_b.clone(), None, None, None, Some(4), Merge::Distance);
        let expected_weight_tile = arr2(&[
            [
                1.0,
                1.0,
                1.0,
                1.0,
                4.0 / 5.0,
                3.0 / 5.0,
                2.0 / 5.0,
                1.0 / 5.0,
            ],
            [
                1.0,
                1.0,
                1.0,
                1.0,
                4.0 / 5.0,
                3.0 / 5.0,
                2.0 / 5.0,
                1.0 / 5.0,
            ],
        ]);
        utils::compare_arrays(
            &weight_tile,
            &expected_weight_tile,
            "Distance weight_tile bc",
        );
    }

    #[test]
    fn test_distance_interpolation_hor_result_abc() {
        let (range_a, tile_a) = small_tile_hor_a();
        let (range_b, tile_b) = small_tile_hor_b();
        let (range_c, tile_c) = small_tile_hor_c();
        let mut tiles = HashMap::<utils::Range2D, utils::Tile>::new();
        let all_ranges = [range_a.clone(), range_b.clone(), range_c.clone()];
        tiles.insert(
            range_a.clone(),
            utils::Tile::new(&range_a.clone(), &tile_a.clone(), &all_ranges),
        );
        tiles.insert(
            range_b.clone(),
            utils::Tile::new(&range_b.clone(), &tile_b.clone(), &all_ranges),
        );
        tiles.insert(
            range_c.clone(),
            utils::Tile::new(&range_c.clone(), &tile_c.clone(), &all_ranges),
        );
        let result = merge_tiles_average_exact(&tiles, None, Merge::Distance);
        let expected_result = arr2(&[
            [
                2.0,
                2.0,
                2.0,
                7.0 / 3.0,
                8.0 / 3.0,
                3.0,
                10.0 / 3.0,
                4.4516134,
                5.6,
                6.4,
                7.2,
                8.0,
                8.0,
                8.0,
                8.0,
            ],
            [
                2.0,
                2.0,
                2.0,
                7.0 / 3.0,
                8.0 / 3.0,
                3.0,
                10.0 / 3.0,
                4.4516134,
                5.6,
                6.4,
                7.2,
                8.0,
                8.0,
                8.0,
                8.0,
            ],
        ]);
        utils::compare_arrays(&result, &expected_result, "Result");
    }
}
