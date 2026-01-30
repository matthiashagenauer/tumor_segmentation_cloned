//! Split full_size into equal sized tiles with overlap.
//!
//! The overlap is
//! - greater than or equal to min_overlap
//! - equal between all tiles within a difference of at most 1
//!
//!
//! Ole-Johan Skrede
//! 12.26.2020
//!

use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Error};
use chrono::Local;
use clap::{value_t, App, Arg};

mod restore;
mod split;
mod utils;

fn main() -> Result<(), Error> {
    let matches = App::new("Extend image orientation")
        .version("0.1.0")
        .author("Ole-Johan Skrede")
        .about("Extends the orientation of multiple images, and writes the result")
        .arg(
            Arg::with_name("input")
                .long("input")
                .value_name("PATH")
                .required(true)
                .help(
                    "Following types of inputs are possible:\n\
                     \t- A single image filename\n\
                     \t- A path to a folder with images somewhere below it\n\
                     \t- A text file with a list of image filepaths\n\
                     \t- A csv file with header ImagePath[,MaskPath]"
                ),
        )
        .arg(
            Arg::with_name("output")
                .long("output")
                .value_name("PATH")
                .required(true)
                .help("Directory to place output. It is created if it does not exist"),
        )
        .arg(
            Arg::with_name("part_size")
                .long("part_size")
                .value_name("INT")
                .default_value("1000")
                .help("Split part size in number of pixels"),
        )
        .arg(
            Arg::with_name("min_overlap")
                .long("min_overlap")
                .value_name("INT")
                .default_value("100")
                .help("Minimum overlap size"),
        )
        .arg(
            Arg::with_name("restore")
                .long("restore")
                .value_name("STR")
                .possible_values(&["simple", "min", "max", "minmax", "uniform", "distance"])
                .help(
                    "If specified, the program collects tiles and tries to restore them to an image. Merging strategies:\n\
                    \t- 'simple': Place tiles where they should be, disregarding if any other tiles overlap\n\
                    \t- 'min': For a pixel with overlapping tiles, assign the smallest pixel value\n\
                    \t- 'max': For a pixel with overlapping tiles, assign the largest pixel value\n\
                    \t- 'minmax': Smooth differences between the 'min' and 'max' option\n\
                    \t- 'uniform': Weighted average of overlapping tiles with uniform weights\n\
                    \t- 'distance': Weighted average of overlapping tiles with weights determined by the\n\
                    \t\tdistance to the non-overlapping region\n\
                    If the input image is color, 'simple' is always chosen."
                ),
        )
        .arg(
            Arg::with_name("align")
                .long("align")
                .help("Align tiles with different overlapping values in restore mode"),
        )
        .arg(
            Arg::with_name("reference")
                .long("reference")
                .value_name("PATH")
                .requires("reference_factor")
                .help(
                    "Only used for restoration. Root path to images corresponding to the result.\n\
                    If a corresponding image is found, resize the result to this images' size\n\
                    before writing."
                ),
        )
        .arg(
            Arg::with_name("reference_factor")
                .long("reference_factor")
                .value_name("FLOAT")
                .help(
                    "Only used for restoration. Expected factor: original size / reference size"
                ),
        )
        .arg(
            Arg::with_name("format")
                .long("format")
                .value_name("STR")
                .possible_values(&["png", "jpg"])
                .default_value("png")
                .help("In which format to write the result"),
        )
        .arg(
            Arg::with_name("debug")
                .long("debug")
                .value_name("PATH")
                .help("If this is given, compute and write debug info to this location."),
        )
        .get_matches();

    println!(
        "[{}] Program start",
        Local::now().format("%d.%m.%Y %H:%M:%S")
    );

    let input = Path::new(matches.value_of("input").unwrap());
    let output = Path::new(matches.value_of("output").unwrap());
    let reference = matches.value_of("reference").map(PathBuf::from);
    let reference_factor = value_t!(matches.value_of("reference_factor"), f32).ok();
    let debug = matches.value_of("debug").map(PathBuf::from);
    let part_size = value_t!(matches.value_of("part_size"), usize)?;
    let min_overlap = value_t!(matches.value_of("min_overlap"), usize)?;
    let output_format = matches.value_of("format").unwrap();

    if !input.exists() {
        return Err(anyhow!("Please specify an existing input path"));
    }

    if input.is_file() && matches.is_present("restore") {
        return Err(anyhow!("Input must be folder when restoring"));
    }

    match matches.value_of("restore") {
        Some(val) => {
            let strategy = if val == "simple" {
                None
            } else {
                Some(restore::restore_utils::Merge::new(val))
            };
            let align = matches.is_present("align");
            let config = restore::Config::new(strategy, align, output_format, reference_factor);
            restore::restore_multiple(input, output, reference, debug, &config)?;
        }
        None => {
            let config = split::Config::new(part_size, min_overlap, output_format);
            let image_paths = if input.is_file() {
                if input.extension() == Some(OsStr::new("txt")) {
                    utils::read_text_file(input)?
                } else if input.extension() == Some(OsStr::new("csv")) {
                    utils::read_csv_file(input)?
                } else {
                    vec![PathBuf::from(input)]
                }
            } else {
                utils::find_image_files(input)
            };
            split::split_multiple(&image_paths, output, &config)?;
        }
    }

    println!(
        "[{}] Program finished",
        Local::now().format("%d.%m.%Y %H:%M:%S")
    );

    Ok(())
}
