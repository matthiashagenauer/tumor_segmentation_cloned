use crate::restore::restore_utils;
use crate::utils;
use image::imageops;
use ndarray::{s, Array2, Zip};
use std::{collections::HashMap, ops::Mul};

fn _sigmoid(x: f32, min: f32, max: f32) -> f32 {
    let scale = 0.2 * (max - min);
    let shift = (min + max) / 2.0;
    (1.0 + ((x - shift) / scale).tanh()) / 2.0
}

fn _circle(radius: usize) -> Array2<u8> {
    let len = 2 * radius + 1;
    let mut in_circle = Array2::<u8>::zeros((len, len));
    let r = radius as i32;
    for (i, x) in (-r..r + 1).enumerate() {
        for (j, y) in (-r..r + 1).enumerate() {
            in_circle[[i, j]] = if x * x + y * y <= r * r { 1 } else { 0 };
        }
    }
    in_circle
}

fn cone(radius: usize) -> Array2<f32> {
    let len = 2 * radius + 1;
    let mut weight_image = Array2::<f32>::zeros((len, len));
    let r = radius as i32;
    for (i, x) in (-r..r + 1).enumerate() {
        for (j, y) in (-r..r + 1).enumerate() {
            let dist_squared = x * x + y * y;
            let dist = (dist_squared as f32).sqrt();
            let x = radius as f32 - dist;
            // let fx = sigmoid(x, 0.0, radius as f32);
            let fx = x * x;
            let weight = if dist_squared > r * r { 0.0 } else { fx };
            weight_image[[i, j]] = weight
        }
    }
    weight_image.clone() / (2.0 / 4.0 * weight_image.sum())
}

fn smooth_difference(min_image: &Array2<u8>, max_image: &Array2<u8>) -> Array2<u8> {
    let diff_threshold = 10;
    let radius = 50;

    let (height, width) = min_image.dim();

    let weight_image = cone(radius);
    let diff_image = max_image - min_image;
    let diff_coords = diff_image
        .indexed_iter()
        .filter(|((_, _), v)| **v > diff_threshold)
        .map(|((i, j), _)| (i, j));

    let mut result_image = Array2::<u8>::zeros(min_image.dim());
    for (i, j) in diff_coords {
        let min_i = (i - radius).max(0);
        let max_i = (i + radius + 1).min(height);
        let min_j = (j - radius).max(0);
        let max_j = (j + radius + 1).min(width);
        let shift_min_i = radius + min_i - i;
        let shift_max_i = radius + max_i - i;
        let shift_min_j = radius + min_j - j;
        let shift_max_j = radius + max_j - j;
        let weight_window =
            weight_image.slice(s![shift_min_i..shift_max_i, shift_min_j..shift_max_j]);
        let weighted_sum = min_image
            .slice(s![min_i..max_i, min_j..max_j])
            .map(|&x| x as f32)
            .mul(&weight_window)
            .sum();
        let result = weighted_sum;
        let result_u8 = (result).clamp(0.0, 255.0) as u8;
        result_image[[i, j]] = result_u8;
    }
    result_image
}

fn resize_with_dim(
    matrix: &Array2<u8>,
    height: u32,
    width: u32,
    filter: imageops::FilterType,
) -> Array2<u8> {
    let orig_im = utils::array2_to_image(matrix);
    let resized_im = imageops::resize(&orig_im, width, height, filter);
    utils::image_to_array2(&resized_im)
}

fn resize_with_factor(
    matrix: &Array2<u8>,
    factor: f32,
    filter: imageops::FilterType,
) -> Array2<u8> {
    let (orig_height, orig_width) = matrix.dim();
    let new_height = (orig_height as f32 * factor).round() as u32;
    let new_width = (orig_width as f32 * factor).round() as u32;
    resize_with_dim(matrix, new_width, new_height, filter)
}

fn matrix_min(a1: &Array2<u8>, a2: &Array2<u8>) -> Array2<u8> {
    assert_eq!(
        a1.shape(),
        a2.shape(),
        "Unequal shape {:?} vs {:?}",
        a1.shape(),
        a2.shape()
    );
    let mut result = Array2::<u8>::zeros(a1.dim());
    Zip::from(&mut result)
        .and(a1)
        .and(a2)
        .for_each(|r, &x1, &x2| *r = x1.min(x2));
    result
}

fn matrix_max(a1: &Array2<u8>, a2: &Array2<u8>) -> Array2<u8> {
    assert_eq!(
        a1.shape(),
        a2.shape(),
        "Unequal shape {:?} vs {:?}",
        a1.shape(),
        a2.shape()
    );
    let mut result = Array2::<u8>::zeros(a1.dim());
    Zip::from(&mut result)
        .and(a1)
        .and(a2)
        .for_each(|r, &x1, &x2| *r = x1.max(x2));
    result
}

pub fn merge_tiles_min(tiles: &HashMap<utils::Range2D, utils::Tile>) -> Array2<u8> {
    let ranges: Vec<utils::Range2D> = tiles.keys().cloned().collect();
    let mut result_image = restore_utils::initial_image(&ranges, 255);

    for (range2d, tile) in tiles {
        let existing_tile = result_image
            .slice(s![range2d.vertical(), range2d.horisontal()])
            .to_owned();
        let result_tile = matrix_min(&tile.image(), &existing_tile);
        result_image
            .slice_mut(s![range2d.vertical(), range2d.horisontal()])
            .assign(&result_tile);
    }
    result_image
}

pub fn merge_tiles_max(tiles: &HashMap<utils::Range2D, utils::Tile>) -> Array2<u8> {
    let ranges: Vec<utils::Range2D> = tiles.keys().cloned().collect();
    let mut result_image = restore_utils::initial_image(&ranges, 0);

    for (range2d, tile) in tiles {
        let existing_tile = result_image
            .slice(s![range2d.vertical(), range2d.horisontal()])
            .to_owned();
        let result_tile = matrix_max(&tile.image(), &existing_tile);
        result_image
            .slice_mut(s![range2d.vertical(), range2d.horisontal()])
            .assign(&result_tile);
    }
    result_image
}

pub fn merge_tiles_minmax(tiles: &HashMap<utils::Range2D, utils::Tile>) -> Array2<u8> {
    let factor = 0.5;
    let min_image = merge_tiles_min(tiles);
    let max_image = merge_tiles_max(tiles);
    let (orig_height, orig_width) = min_image.dim();
    let small_min_image = resize_with_factor(&min_image, factor, imageops::FilterType::CatmullRom);
    let small_max_image = resize_with_factor(&max_image, factor, imageops::FilterType::CatmullRom);
    let small_smooth_image = smooth_difference(&small_min_image, &small_max_image);
    let smooth_image = resize_with_dim(
        &small_smooth_image,
        orig_height as u32,
        orig_width as u32,
        imageops::FilterType::CatmullRom,
    );
    let mut result = Array2::<u8>::zeros(min_image.dim());
    Zip::from(&mut result)
        .and(&min_image)
        .and(&smooth_image)
        .for_each(|r, &x, &y| *r = x.max(y));
    result
    // smooth_image
}
