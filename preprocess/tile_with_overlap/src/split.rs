use anyhow::{anyhow, Error};
use image::{io::Reader, GenericImageView};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    fs::create_dir_all,
    ops::Range,
    path::{Path, PathBuf},
};

use crate::utils;

pub struct Config {
    part_size: usize,
    min_overlap: usize,
    output_format: String,
}

impl Config {
    pub fn new(part_size: usize, min_overlap: usize, output_format: &str) -> Self {
        Config {
            part_size,
            min_overlap,
            output_format: String::from(output_format),
        }
    }

    pub fn part_size(&self) -> usize {
        self.part_size
    }

    pub fn min_overlap(&self) -> usize {
        self.min_overlap
    }

    pub fn output_format(&self) -> String {
        self.output_format.clone()
    }
}

fn find_overlap(full_size: usize, part_size: usize, min_overlap: usize) -> (f64, usize) {
    assert!(full_size > part_size, "Part is greater than the whole");
    let mut num_parts = (full_size as f64 / part_size as f64).ceil() as usize;
    let mut overlap = (part_size * num_parts - full_size) as f64 / ((num_parts - 1) as f64);
    if overlap < min_overlap as f64 {
        assert!(
            part_size > min_overlap,
            "Part size {} must be greater than min_overlap {}",
            part_size,
            min_overlap
        );
        num_parts =
            ((full_size - min_overlap) as f64 / (part_size - min_overlap) as f64).ceil() as usize;
        overlap = (part_size * num_parts - full_size) as f64 / ((num_parts - 1) as f64);
    }
    (overlap, num_parts)
}

fn divide_with_overlap(
    full_size: usize,
    part_size: usize,
    min_overlap: usize,
) -> Vec<Range<usize>> {
    let mut ranges = Vec::<Range<usize>>::new();
    if full_size > part_size {
        let (overlap, num_parts) = find_overlap(full_size, part_size, min_overlap);
        let fractional_part = overlap - overlap.floor();
        let num_ceils = (num_parts as f64 * fractional_part).floor() as usize;
        for k in 0..num_parts {
            let int_overlap = if k <= num_ceils {
                overlap.ceil() as usize
            } else {
                overlap.floor() as usize
            };
            let start = if k == 0 {
                0
            } else {
                ranges[k - 1].end - int_overlap
            };
            let end = start + part_size;
            ranges.push(start..end);
        }
    } else {
        ranges.push(0..full_size)
    }

    ranges
}

fn split_image(
    image: &image::DynamicImage,
    part_size: usize,
    min_overlap: usize,
) -> HashMap<utils::Range2D, image::DynamicImage> {
    let (width, height) = image.dimensions();
    let vertical_ranges = divide_with_overlap(height as usize, part_size, min_overlap);
    let horisontal_ranges = divide_with_overlap(width as usize, part_size, min_overlap);
    let mut tiles_and_location = HashMap::<utils::Range2D, image::DynamicImage>::new();
    for vertical_range in vertical_ranges.iter() {
        for horisontal_range in horisontal_ranges.iter() {
            let tile = image.crop_imm(
                horisontal_range.start as u32,
                vertical_range.start as u32,
                part_size as u32,
                part_size as u32,
            );
            let range_2d = utils::Range2D::new(vertical_range.clone(), horisontal_range.clone());
            tiles_and_location.insert(range_2d, tile);
        }
    }
    tiles_and_location
}

pub fn split_single(input: &Path, output: &Path, config: &Config) -> Result<(), Error> {
    let mut input_reader = Reader::open(input)?;
    input_reader.no_limits();
    let input_image = input_reader.decode()?;
    let name_base = input.file_stem().unwrap().to_str().unwrap();
    let tiles_and_location = split_image(&input_image, config.part_size(), config.min_overlap());
    create_dir_all(output)?;
    for (range_2d, tile) in tiles_and_location.iter() {
        let name = format!(
            "{}_rows-{:05}-{:05}_cols-{:05}-{:05}.{}",
            name_base,
            range_2d.vertical().start,
            range_2d.vertical().end,
            range_2d.horisontal().start,
            range_2d.horisontal().end,
            config.output_format(),
        );
        tile.save(output.join(name))?;
    }

    Ok(())
}

pub fn split_multiple(
    image_paths: &[PathBuf],
    output_root: &Path,
    config: &Config,
) -> Result<(), Error> {
    let common_path =
        utils::common_path(image_paths).ok_or_else(|| anyhow!("Found no common path"))?;
    let pbar = ProgressBar::new(image_paths.len() as u64);
    pbar.set_style(ProgressStyle::default_bar().template(
        "[{elapsed} elapsed] {wide_bar:.cyan/white} {percent}% [{eta} remaining] [rendering]",
    )?);
    image_paths.par_iter().for_each(|image_path| {
        let relative = image_path.strip_prefix(common_path.clone()).unwrap();
        let output = output_root.join(relative.with_extension(""));
        match split_single(image_path, &output, config) {
            Ok(_) => {}
            Err(e) => eprintln!("\n{}:\n{:?}", image_path.display(), e),
        }
        pbar.inc(1);
    });
    pbar.finish();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{divide_with_overlap, split_image};
    use crate::utils;

    #[test]
    // full_size is smaller than part_size
    fn test_division_1() {
        let ranges = divide_with_overlap(10, 20, 0);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 0..10);
    }

    #[test]
    // full_size is smaller equal to part_size
    fn test_division_2() {
        let ranges = divide_with_overlap(10, 10, 0);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 0..10);
    }

    #[test]
    // full_size is greater than part_size with 2 parts and 8 overlap
    fn test_division_3() {
        let ranges = divide_with_overlap(12, 10, 0);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0], 0..10);
        assert_eq!(ranges[1], 2..12);
    }

    #[test]
    // full_size is greater than part_size with 2 parts and 0 overlap
    fn test_division_4() {
        let ranges = divide_with_overlap(20, 10, 0);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0], 0..10);
        assert_eq!(ranges[1], 10..20);
    }

    #[test]
    // full_size is greater than part_size and min_overlap=3 with 3 parts and 5 overlap
    fn test_division_5() {
        let ranges = divide_with_overlap(20, 10, 3);
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0], 0..10);
        assert_eq!(ranges[1], 5..15);
        assert_eq!(ranges[2], 10..20);
    }

    #[test]
    // full_size is greater than part_size and min_overlap=3 with 3 parts and 6 and 5 overlap
    fn test_division_6() {
        let ranges = divide_with_overlap(19, 10, 3);
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0], 0..10);
        assert_eq!(ranges[1], 4..14);
        assert_eq!(ranges[2], 9..19);
    }

    #[test]
    #[should_panic]
    /// Should panic because min_overlap is greater than part_size
    fn test_division_7() {
        let _ = divide_with_overlap(10, 7, 8);
    }

    #[test]
    fn test_split() {
        let tile_11 = utils::test_tile_11();
        let tile_12 = utils::test_tile_12();
        let tile_13 = utils::test_tile_13();
        let tile_21 = utils::test_tile_21();
        let tile_22 = utils::test_tile_22();
        let tile_23 = utils::test_tile_23();
        let test_image =
            image::DynamicImage::ImageLuma8(utils::array2_to_image(&utils::test_image_exact()));
        let result = split_image(&test_image, 8, 3);
        let result_ranges: Vec<utils::Range2D> = result.keys().cloned().collect();

        assert_eq!(result.len(), 6, "Wrong number of tiles");

        assert!(result_ranges.contains(&tile_11.0));
        assert!(result_ranges.contains(&tile_12.0));
        assert!(result_ranges.contains(&tile_13.0));
        assert!(result_ranges.contains(&tile_21.0));
        assert!(result_ranges.contains(&tile_22.0));
        assert!(result_ranges.contains(&tile_23.0));

        assert_eq!(
            result.get(&tile_11.0).unwrap().as_luma8().unwrap(),
            &utils::array2_to_image(&tile_11.1)
        );
        assert_eq!(
            result.get(&tile_12.0).unwrap().as_luma8().unwrap(),
            &utils::array2_to_image(&tile_12.1)
        );
        assert_eq!(
            result.get(&tile_13.0).unwrap().as_luma8().unwrap(),
            &utils::array2_to_image(&tile_13.1)
        );
        assert_eq!(
            result.get(&tile_21.0).unwrap().as_luma8().unwrap(),
            &utils::array2_to_image(&tile_21.1)
        );
        assert_eq!(
            result.get(&tile_22.0).unwrap().as_luma8().unwrap(),
            &utils::array2_to_image(&tile_22.1)
        );
        assert_eq!(
            result.get(&tile_23.0).unwrap().as_luma8().unwrap(),
            &utils::array2_to_image(&tile_23.1)
        );
    }
}
