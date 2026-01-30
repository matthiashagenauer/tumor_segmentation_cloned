use crate::utils;
use anyhow::{anyhow, Error};
use image::{imageops, DynamicImage, GenericImageView, GrayImage};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use ndarray::{s, Array2};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    fs::create_dir_all,
    path::{Path, PathBuf},
};

pub mod align_tiles;
pub mod merge_average;
pub mod merge_minmax;
pub mod restore_utils;
pub mod simple;

pub struct Config {
    strategy: Option<restore_utils::Merge>,
    align: bool,
    output_format: String,
    exact: bool,
    reference_factor: Option<f32>,
}

impl Config {
    pub fn new(
        strategy: Option<restore_utils::Merge>,
        align: bool,
        output_format: &str,
        reference_factor: Option<f32>,
    ) -> Self {
        Config {
            strategy,
            align,
            output_format: String::from(output_format),
            exact: true,
            reference_factor,
        }
    }

    pub fn strategy(&self) -> Option<restore_utils::Merge> {
        self.strategy
    }

    pub fn align(&self) -> bool {
        self.align
    }

    pub fn output_format(&self) -> String {
        self.output_format.clone()
    }

    pub fn exact(&self) -> bool {
        self.exact
    }

    pub fn reference_factor(&self) -> Option<f32> {
        self.reference_factor
    }
}

/// Input path is expected to be on the form
///
/// /path/to/<case_id>/<case_id>_rows-<v_start>-<v_end>_cols-<h_start>-<h_end>.<ext>
///
/// The filename must start with the name of its parent folder, and v_start, v_end, h_start, h_end
/// are zero-padded integers representing the inclusive vertical and horisontal start pixels
/// coordinates, and the exclusive vertical and horisontal end pixels.
///
fn parse_name(path: &Path, case_id: &str) -> Result<utils::Range2D, Error> {
    let file_stem = path.file_stem().unwrap().to_str().unwrap();
    assert!(
        file_stem.starts_with(case_id),
        "Invalid file name: {} from case {}",
        file_stem,
        case_id
    );
    let last_part = file_stem.replace(format!("{}_", &case_id).as_str(), "");
    let parts: Vec<&str> = last_part.split('_').collect();
    assert!(parts.len() == 2, "Invalid format on file name {:?}", &path);
    let v_parts: Vec<&str> = parts[0].split('-').collect();
    let h_parts: Vec<&str> = parts[1].split('-').collect();
    let v_start = v_parts[1];
    let v_end = v_parts[2];
    let h_start = h_parts[1];
    let h_end = h_parts[2];
    let v_range = v_start.parse()?..v_end.parse()?;
    let h_range = h_start.parse()?..h_end.parse()?;
    Ok(utils::Range2D::new(v_range, h_range))
}

fn nearest_element_in_list(value: u8, list: &[u8]) -> u8 {
    let mut min_dist = 255;
    let mut min_element = list[0];
    for element in list {
        let dist = if *element >= value {
            element - value
        } else {
            value - element
        };
        if dist < min_dist {
            min_dist = dist;
            min_element = *element;
        }
    }
    min_element
}

/// The rescaling tends to involve interpolation of values. This means that an input mask with
/// values, say {0, 255}, will get distorted such that the rescaled mask may have values, say {0,
/// 1, 2, 253, 254, 255}. This funciton should clean the image, and guarantee that it only contains
/// its original values.
fn clean_grayscale(original_image: &DynamicImage, image_buf: &GrayImage) -> GrayImage {
    let original_buf = original_image.to_luma8();
    let unique: Vec<u8> = original_buf.pixels().map(|p| p[0]).unique().collect();
    let mut image_buf = image_buf.clone();
    for pixel in image_buf.pixels_mut() {
        *pixel = image::Luma([nearest_element_in_list(pixel[0], &unique)]);
    }
    image_buf
}

fn collect_tiles(
    input: &Path,
    maybe_global_target_dim: Option<(usize, usize)>,
    maybe_reference_factor: Option<f32>,
) -> Result<
    (
        HashMap<utils::Range2D, image::DynamicImage>,
        image::ColorType,
    ),
    Error,
> {
    let case_id = input
        .file_stem()
        .ok_or_else(|| anyhow!("Input folder name"))?
        .to_str()
        .ok_or_else(|| anyhow!("osstr to str"))?;
    let mut paths_and_location = HashMap::<utils::Range2D, PathBuf>::new();
    for entry in (input.read_dir()?).flatten() {
        let path = entry.path();
        let file_name = path.file_name().unwrap().to_str().unwrap();
        if path.is_file()
            && path.extension().is_some()
            && utils::IMAGE_EXTENSIONS.contains(&path.extension().unwrap().to_str().unwrap())
        {
            if let Some(index) = file_name.find("_rows-") {
                let case_id_candidate = file_name.get(0..index).unwrap();
                assert_eq!(&case_id, &case_id_candidate, "Multiple cases in folder");
                let range2d = parse_name(&path, case_id)?;
                paths_and_location.insert(range2d, path);
            }
        }
    }

    if let Some(global_target_dim) = maybe_global_target_dim {
        let ranges: Vec<utils::Range2D> = paths_and_location.keys().cloned().collect();
        let (global_height, global_width) = restore_utils::global_dim_from_ranges(&ranges)
            .ok_or_else(|| {
                anyhow!(
                    "ERROR: Extract global dim from ranges in {}",
                    input.display()
                )
            })?;
        if global_height != global_target_dim.0 || global_width != global_target_dim.1 {
            let resize_factor = match maybe_reference_factor {
                Some(v) => v,
                None => return Err(anyhow!("Expected reference resize factor")),
            };
            let v_factor = 1.0 / resize_factor;
            let h_factor = 1.0 / resize_factor;
            let mut new_paths_and_location = HashMap::<utils::Range2D, PathBuf>::new();
            for (range, path) in paths_and_location {
                let new_range = range.scale(h_factor, v_factor);
                new_paths_and_location.insert(new_range, path);
            }
            paths_and_location = new_paths_and_location;
        }
    }

    let mut tiles_and_location = HashMap::<utils::Range2D, image::DynamicImage>::new();
    let mut color_type: Option<image::ColorType> = None;
    for (range, path) in paths_and_location {
        let mut im = image::open(&path)?;
        let target_height = range.height() as u32;
        let target_width = range.width() as u32;
        let (current_width, current_height) = im.dimensions();
        if (current_height != target_height) || (current_width != target_width) {
            let resized_im = im.resize_exact(
                target_width,
                target_height,
                imageops::FilterType::CatmullRom,
            );
            if resized_im.color() == image::ColorType::L8 {
                im = DynamicImage::ImageLuma8(clean_grayscale(&im, &resized_im.to_luma8()))
            } else {
                im = resized_im;
            }
        }
        let temp_color_type = im.color();
        match color_type {
            Some(val) => assert_eq!(val, temp_color_type, "Different color type in tiles"),
            None => color_type = Some(temp_color_type),
        }
        tiles_and_location.insert(range, im);
    }
    Ok((tiles_and_location, color_type.unwrap()))
}

fn merge_tiles(
    tiles: &HashMap<utils::Range2D, utils::Tile>,
    debug_dir: Option<PathBuf>,
    strategy: restore_utils::Merge,
    exact: bool,
) -> Array2<u8> {
    match strategy {
        restore_utils::Merge::Min => merge_minmax::merge_tiles_min(tiles),
        restore_utils::Merge::Max => merge_minmax::merge_tiles_max(tiles),
        restore_utils::Merge::MinMax => merge_minmax::merge_tiles_minmax(tiles),
        _ => {
            if exact {
                merge_average::merge_tiles_average_exact(tiles, debug_dir, strategy)
                    .map(|v| v.round() as u8)
            } else {
                merge_average::merge_tiles_average_round(tiles, debug_dir, strategy)
            }
        }
    }
}

fn final_resize(
    ranges: &[utils::Range2D],
    restored_image: &Array2<u8>,
    reference_height: usize,
    reference_width: usize,
) -> Array2<u8> {
    let fill_value = 0_u8;
    let mut padded_image = Array2::ones((reference_height, reference_width)) * fill_value;

    let min_row = 0;
    let min_col = 0;

    let max_row = restore_utils::ranges_max_row(ranges).unwrap();
    let max_col = restore_utils::ranges_max_col(ranges).unwrap();

    let max_row = max_row.min(reference_height);
    let max_col = max_col.min(reference_width);

    let restored_image = restored_image.slice(s![
        0..max_row.min(restored_image.shape()[0]),
        0..max_col.min(restored_image.shape()[1])
    ]);

    let max_row = min_row + restored_image.shape()[0];
    let max_col = min_col + restored_image.shape()[1];

    padded_image
        .slice_mut(s![min_row..max_row, min_col..max_col,])
        .assign(&restored_image);

    padded_image
}

/// Restore an image from tiles found directly bottom input folder. Subimages are stored with a file
/// name on the form
///
/// /path/to/<case_id>/<case_id>_rows-<v_start>-<v_end>_cols-<h_start>-<h_end>.<output_format>
///
/// Output is written in the output folder with filename <case_id>.<output_format>
fn restore_single(
    input: &Path,
    output_path: &Path,
    reference_path: Option<PathBuf>,
    debug_dir: Option<PathBuf>,
    config: &Config,
) -> Result<(), Error> {
    if debug_dir.is_some() {
        println!("Reading tile folder {}", input.display());
    }
    assert!(input.is_dir(), "Input must be directory");

    let maybe_dim = reference_path
        .and_then(|p| image::image_dimensions(p).ok())
        .map(|(x, y)| (y as usize, x as usize));

    let (image_tiles_and_location, color_type) =
        collect_tiles(input, maybe_dim, config.reference_factor())?;
    if debug_dir.is_some() {
        println!("Collected {} tiles", image_tiles_and_location.len());
    }
    if image_tiles_and_location.is_empty() {
        return Err(anyhow!("Could not find any tiles in {}", input.display()));
    }
    let restored_image = match config.strategy() {
        None => simple::restore_simple(&image_tiles_and_location, color_type)?,
        Some(merge_strategy) => {
            if color_type != image::ColorType::L8 {
                return Err(anyhow!(
                    "Only grayscale tiles can be merged. Use 'simple' strategy for color tiles"
                ));
            }
            let all_ranges: Vec<utils::Range2D> =
                image_tiles_and_location.keys().cloned().collect();
            let mut tiles = HashMap::<utils::Range2D, utils::Tile>::new();
            for (range, im) in image_tiles_and_location {
                tiles.insert(
                    range.clone(),
                    utils::Tile::new(&range, &utils::image_to_array2(&im.to_luma8()), &all_ranges),
                );
            }
            if config.align() {
                let down_threshold = 0.3;
                let centre_threshold = 0.5;
                tiles = align_tiles::align(&tiles, down_threshold, centre_threshold);
            }
            let mut merged_image = merge_tiles(&tiles, debug_dir, merge_strategy, config.exact());
            if let Some((h, w)) = maybe_dim {
                merged_image = final_resize(&all_ranges, &merged_image, h, w);
            }
            image::DynamicImage::ImageLuma8(utils::array2_to_image(&merged_image))
        }
    };

    create_dir_all(output_path.parent().expect("parent"))?;
    restored_image.save(output_path)?;
    Ok(())
}

pub fn restore_multiple(
    input_root: &Path,
    output_root: &Path,
    reference_root: Option<PathBuf>,
    debug_root: Option<PathBuf>,
    config: &Config,
) -> Result<(), Error> {
    let image_folders = utils::find_image_folders(input_root);
    println!("Found {} tile folder{}", image_folders.len(), utils::plural_s(image_folders.len()));
    let mut pbar = ProgressBar::new(image_folders.len() as u64);
    pbar.set_style(ProgressStyle::default_bar().template(
        "[{elapsed} elapsed] {wide_bar:.cyan/white} {percent}% [{eta} remaining] [rendering]",
    )?);
    if image_folders.len() == 1 {
        pbar = ProgressBar::hidden();
    }
    image_folders.par_iter().for_each(|image_folder| {
        let relative = image_folder.strip_prefix(input_root).unwrap();
        let output_path = if image_folders.len() == 1 && output_root.extension().is_some() {
            output_root.to_path_buf()
        } else {
            let temp = output_root.join(relative);
            let output_dir = if relative == Path::new("") {
                output_root
            } else {
                temp.parent().unwrap()
            };
            let case_id = image_folder
                .file_stem()
                .expect("Input folder name")
                .to_str()
                .expect("osstr to str");
            output_dir.join(format!("{}.{}", case_id, config.output_format()))
        };
        if !output_path.exists() {
            let reference_path = match reference_root {
                Some(ref root) => {
                    let rel_dir = root.join(relative);
                    let parent_dir = rel_dir.parent().expect("parent of reference");
                    let stem = rel_dir
                        .file_stem()
                        .expect("file name")
                        .to_str()
                        .expect("osstr to str");
                    let path = parent_dir.join(format!("{}.png", stem));
                    if path.exists() {
                        Some(path)
                    } else if parent_dir.join(format!("{}/{}.png", stem, stem)).exists() {
                        Some(parent_dir.join(format!("{}/{}.png", stem, stem)))
                    } else {
                        println!(
                            "WARNING: Expected reference path does not exist: {}",
                            path.display()
                        );
                        None
                    }
                }
                None => None,
            };
            let debug_dir = match debug_root {
                Some(ref root) => {
                    let path = root.join(relative);
                    if !path.exists() {
                        match create_dir_all(&path) {
                            Ok(_) => Some(path),
                            Err(_) => None,
                        }
                    } else {
                        Some(path)
                    }
                }
                None => None,
            };

            match restore_single(
                image_folder,
                &output_path,
                reference_path,
                debug_dir,
                config,
            ) {
                Ok(_) => {}
                Err(e) => eprintln!("Error processing {}:\n{:?}", image_folder.display(), e),
            }
        }
        pbar.inc(1);
    });
    pbar.finish();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{merge_tiles, restore_utils::Merge};
    use crate::utils;
    use std::collections::HashMap;

    #[test]
    fn test_merge_exact() {
        let (range_11, tile_11) = utils::test_tile_11();
        let (range_12, tile_12) = utils::test_tile_12();
        let (range_13, tile_13) = utils::test_tile_13();
        let (range_21, tile_21) = utils::test_tile_21();
        let (range_22, tile_22) = utils::test_tile_22();
        let (range_23, tile_23) = utils::test_tile_23();
        let mut tiles = HashMap::<utils::Range2D, utils::Tile>::new();
        let all_ranges = [
            range_11.clone(),
            range_12.clone(),
            range_13.clone(),
            range_21.clone(),
            range_22.clone(),
            range_23.clone(),
        ];
        tiles.insert(
            range_11.clone(),
            utils::Tile::new(&range_11.clone(), &tile_11.clone(), &all_ranges),
        );
        tiles.insert(
            range_12.clone(),
            utils::Tile::new(&range_12.clone(), &tile_12.clone(), &all_ranges),
        );
        tiles.insert(
            range_13.clone(),
            utils::Tile::new(&range_13.clone(), &tile_13.clone(), &all_ranges),
        );
        tiles.insert(
            range_21.clone(),
            utils::Tile::new(&range_21.clone(), &tile_21.clone(), &all_ranges),
        );
        tiles.insert(
            range_22.clone(),
            utils::Tile::new(&range_22.clone(), &tile_22.clone(), &all_ranges),
        );
        tiles.insert(
            range_23.clone(),
            utils::Tile::new(&range_23.clone(), &tile_23.clone(), &all_ranges),
        );
        let merged_image = merge_tiles(&tiles, None, Merge::Uniform, true);
        assert_eq!(merged_image, utils::test_image_exact())
    }

    #[test]
    fn test_merge_round() {
        let (range_11, tile_11) = utils::test_tile_11();
        let (range_12, tile_12) = utils::test_tile_12();
        let (range_13, tile_13) = utils::test_tile_13();
        let (range_21, tile_21) = utils::test_tile_21();
        let (range_22, tile_22) = utils::test_tile_22();
        let (range_23, tile_23) = utils::test_tile_23();
        let mut tiles = HashMap::<utils::Range2D, utils::Tile>::new();
        let all_ranges = [
            range_11.clone(),
            range_12.clone(),
            range_13.clone(),
            range_21.clone(),
            range_22.clone(),
            range_23.clone(),
        ];
        tiles.insert(
            range_11.clone(),
            utils::Tile::new(&range_11.clone(), &tile_11.clone(), &all_ranges),
        );
        tiles.insert(
            range_12.clone(),
            utils::Tile::new(&range_12.clone(), &tile_12.clone(), &all_ranges),
        );
        tiles.insert(
            range_13.clone(),
            utils::Tile::new(&range_13.clone(), &tile_13.clone(), &all_ranges),
        );
        tiles.insert(
            range_21.clone(),
            utils::Tile::new(&range_21.clone(), &tile_21.clone(), &all_ranges),
        );
        tiles.insert(
            range_22.clone(),
            utils::Tile::new(&range_22.clone(), &tile_22.clone(), &all_ranges),
        );
        tiles.insert(
            range_23.clone(),
            utils::Tile::new(&range_23.clone(), &tile_23.clone(), &all_ranges),
        );
        // let merged_image = utils::image_to_array2(&merge_tiles(&tiles_and_location));
        let merged_image = merge_tiles(&tiles, None, Merge::Uniform, false);
        assert_eq!(merged_image, utils::test_image_round())
    }
}
