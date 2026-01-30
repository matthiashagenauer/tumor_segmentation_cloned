use crate::utils;
use anyhow::{anyhow, Error};
use std::{
    collections::HashMap,
    ops::{Deref, DerefMut, Range},
};

fn put_window<P, Container>(
    im: &mut image::ImageBuffer<P, Container>,
    window: &image::ImageBuffer<P, Container>,
    rows: Range<usize>,
    cols: Range<usize>,
) where
    P: image::Pixel + 'static,
    P::Subpixel: 'static,
    Container: Deref<Target = [P::Subpixel]> + DerefMut,
{
    assert_eq!(rows.len(), window.height() as usize);
    assert_eq!(cols.len(), window.width() as usize);
    assert!(rows.end <= im.height() as usize);
    assert!(cols.end <= im.width() as usize);
    for (loc_i, glob_i) in rows.enumerate() {
        for (loc_j, glob_j) in cols.clone().enumerate() {
            im.put_pixel(
                glob_j as u32,
                glob_i as u32,
                *window.get_pixel(loc_j as u32, loc_i as u32),
            );
        }
    }
}

pub fn restore_simple(
    tiles_and_location: &HashMap<utils::Range2D, image::DynamicImage>,
    color_type: image::ColorType,
) -> Result<image::DynamicImage, Error> {
    let ranges: Vec<utils::Range2D> = tiles_and_location.keys().cloned().collect();
    let height = ranges.iter().map(|r| r.vertical().end).max().unwrap();
    let width = ranges.iter().map(|r| r.horisontal().end).max().unwrap();

    match color_type {
        image::ColorType::L8 => {
            let mut result_image = image::GrayImage::new(width as u32, height as u32);
            for (range2d, tile) in tiles_and_location {
                match tile {
                    image::DynamicImage::ImageLuma8(buf) => {
                        put_window(
                            &mut result_image,
                            buf,
                            range2d.vertical(),
                            range2d.horisontal(),
                        );
                    }
                    _ => unreachable!(),
                }
            }
            Ok(image::DynamicImage::ImageLuma8(result_image))
        }
        image::ColorType::Rgb8 => {
            let mut result_image = image::RgbImage::new(width as u32, height as u32);
            for (range2d, tile) in tiles_and_location {
                match tile {
                    image::DynamicImage::ImageRgb8(buf) => {
                        put_window(
                            &mut result_image,
                            buf,
                            range2d.vertical(),
                            range2d.horisontal(),
                        );
                    }
                    _ => unreachable!(),
                }
            }
            Ok(image::DynamicImage::ImageRgb8(result_image))
        }
        _ => Err(anyhow!("Unimplemented color type {:?}", color_type)),
    }
}
