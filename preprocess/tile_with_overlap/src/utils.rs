use anyhow::{anyhow, Error};
use ndarray::{arr2, Array2};
use serde::Deserialize;
use std::{
    cmp::Ordering,
    collections::HashSet,
    fmt,
    fs::File,
    io::{BufRead, BufReader},
    ops::Range,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

pub const IMAGE_EXTENSIONS: [&str; 3] = ["png", "jpg", "jpeg"];

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Range2D {
    vertical: Range<usize>,
    horisontal: Range<usize>,
}

impl Range2D {
    pub fn new(vertical: Range<usize>, horisontal: Range<usize>) -> Range2D {
        Range2D {
            vertical,
            horisontal,
        }
    }

    pub fn vertical(&self) -> Range<usize> {
        self.vertical.clone()
    }

    pub fn horisontal(&self) -> Range<usize> {
        self.horisontal.clone()
    }

    pub fn height(&self) -> usize {
        self.vertical.len()
    }

    pub fn width(&self) -> usize {
        self.horisontal.len()
    }

    pub fn top(&self) -> usize {
        self.vertical.start
    }

    pub fn bottom(&self) -> usize {
        self.vertical.end
    }

    pub fn left(&self) -> usize {
        self.horisontal.start
    }

    pub fn right(&self) -> usize {
        self.horisontal.end
    }

    pub fn scale(&self, v_factor: f32, h_factor: f32) -> Self {
        let new_v_start = (self.vertical.start as f32 * v_factor).floor() as usize;
        let new_v_end = (self.vertical.end as f32 * v_factor).floor() as usize;
        let new_h_start = (self.horisontal.start as f32 * h_factor).floor() as usize;
        let new_h_end = (self.horisontal.end as f32 * h_factor).floor() as usize;
        Range2D {
            vertical: new_v_start..new_v_end,
            horisontal: new_h_start..new_h_end,
        }
    }

    pub fn _shift(&self, v_delta: isize, h_delta: isize) -> Self {
        let new_v_start = (self.vertical.start as isize + v_delta) as usize;
        let new_v_end = (self.vertical.end as isize + v_delta) as usize;
        let new_h_start = (self.horisontal.start as isize + h_delta) as usize;
        let new_h_end = (self.horisontal.end as isize + h_delta) as usize;
        Range2D {
            vertical: new_v_start..new_v_end,
            horisontal: new_h_start..new_h_end,
        }
    }
}

impl Ord for Range2D {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.top() == other.top() {
            self.left().cmp(&other.left())
        } else {
            self.top().cmp(&other.top())
        }
    }
}

impl PartialOrd for Range2D {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Range2D {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "row-{:05}-{:05}-col-{:05}-{:05}",
            self.top(),
            self.bottom(),
            self.left(),
            self.right()
        )
    }
}

#[derive(Clone)]
pub struct Tile {
    range2d: Range2D,
    image: Array2<u8>,
    overlapping_tiles_top: Vec<Range2D>,
    overlapping_tiles_bottom: Vec<Range2D>,
    overlapping_tiles_left: Vec<Range2D>,
    overlapping_tiles_right: Vec<Range2D>,
    overlap_ranges_top: Vec<Range<usize>>,
    overlap_ranges_bottom: Vec<Range<usize>>,
    overlap_ranges_left: Vec<Range<usize>>,
    overlap_ranges_right: Vec<Range<usize>>,
}

impl Tile {
    pub fn new(range2d: &Range2D, image: &Array2<u8>, other_ranges: &[Range2D]) -> Self {
        let mut overlapping_tiles_top = Vec::<Range2D>::new();
        let mut overlapping_tiles_bottom = Vec::<Range2D>::new();
        let mut overlapping_tiles_left = Vec::<Range2D>::new();
        let mut overlapping_tiles_right = Vec::<Range2D>::new();
        let mut overlap_ranges_top = Vec::<Range<usize>>::new();
        let mut overlap_ranges_bottom = Vec::<Range<usize>>::new();
        let mut overlap_ranges_left = Vec::<Range<usize>>::new();
        let mut overlap_ranges_right = Vec::<Range<usize>>::new();
        for other_range2d in other_ranges.iter() {
            if range2d == other_range2d || !tiles_overlap(range2d, other_range2d) {
                continue;
            } else {
                if let Some(range) = overlap_top(range2d, other_range2d) {
                    overlap_ranges_top.push(range.clone());
                    overlapping_tiles_top.push(other_range2d.clone());
                }
                if let Some(range) = overlap_bottom(range2d, other_range2d) {
                    overlap_ranges_bottom.push(range.clone());
                    overlapping_tiles_bottom.push(other_range2d.clone());
                }
                if let Some(range) = overlap_left(range2d, other_range2d) {
                    overlap_ranges_left.push(range.clone());
                    overlapping_tiles_left.push(other_range2d.clone());
                }
                if let Some(range) = overlap_right(range2d, other_range2d) {
                    overlap_ranges_right.push(range.clone());
                    overlapping_tiles_right.push(other_range2d.clone());
                }
            }
        }

        Tile {
            range2d: range2d.clone(),
            image: image.clone(),
            overlapping_tiles_top,
            overlapping_tiles_bottom,
            overlapping_tiles_left,
            overlapping_tiles_right,
            overlap_ranges_top,
            overlap_ranges_bottom,
            overlap_ranges_left,
            overlap_ranges_right,
        }
    }

    pub fn range2d(&self) -> Range2D {
        self.range2d.clone()
    }

    pub fn image(&self) -> Array2<u8> {
        self.image.clone()
    }

    pub fn _overlapping_tiles_top(&self) -> Vec<Range2D> {
        self.overlapping_tiles_top.clone()
    }

    pub fn _overlapping_tiles_bottom(&self) -> Vec<Range2D> {
        self.overlapping_tiles_bottom.clone()
    }

    pub fn _overlapping_tiles_left(&self) -> Vec<Range2D> {
        self.overlapping_tiles_left.clone()
    }

    pub fn _overlapping_tiles_right(&self) -> Vec<Range2D> {
        self.overlapping_tiles_right.clone()
    }

    pub fn overlapping_tiles_direct_top(&self) -> Vec<Range2D> {
        let mut candidates = Vec::<Range2D>::new();
        for range2d in self.overlapping_tiles_top.iter() {
            if self.range2d.horisontal() == range2d.horisontal() {
                candidates.push(range2d.clone());
            }
        }
        candidates
    }

    pub fn overlapping_tiles_direct_bottom(&self) -> Vec<Range2D> {
        let mut candidates = Vec::<Range2D>::new();
        for range2d in self.overlapping_tiles_bottom.iter() {
            if self.range2d.horisontal() == range2d.horisontal() {
                candidates.push(range2d.clone());
            }
        }
        candidates
    }

    pub fn overlapping_tiles_direct_left(&self) -> Vec<Range2D> {
        let mut candidates = Vec::<Range2D>::new();
        for range2d in self.overlapping_tiles_left.iter() {
            if self.range2d.vertical() == range2d.vertical() {
                candidates.push(range2d.clone());
            }
        }
        candidates
    }

    pub fn overlapping_tiles_direct_right(&self) -> Vec<Range2D> {
        let mut candidates = Vec::<Range2D>::new();
        for range2d in self.overlapping_tiles_right.iter() {
            if self.range2d.vertical() == range2d.vertical() {
                candidates.push(range2d.clone());
            }
        }
        candidates
    }

    pub fn max_overlap_top(&self) -> Option<Range<usize>> {
        self.overlap_ranges_top
            .iter()
            .max_by_key(|r| r.len())
            .cloned()
    }

    pub fn max_overlap_bottom(&self) -> Option<Range<usize>> {
        self.overlap_ranges_bottom
            .iter()
            .max_by_key(|r| r.len())
            .cloned()
    }

    pub fn max_overlap_left(&self) -> Option<Range<usize>> {
        self.overlap_ranges_left
            .iter()
            .max_by_key(|r| r.len())
            .cloned()
    }

    pub fn max_overlap_right(&self) -> Option<Range<usize>> {
        self.overlap_ranges_right
            .iter()
            .max_by_key(|r| r.len())
            .cloned()
    }

    pub fn max_overlapping_tile_direct_top(&self) -> Option<Range2D> {
        self.overlapping_tiles_direct_top()
            .iter()
            .max_by_key(|r| r.vertical().len())
            .cloned()
    }

    pub fn max_overlapping_tile_direct_bottom(&self) -> Option<Range2D> {
        self.overlapping_tiles_direct_bottom()
            .iter()
            .max_by_key(|r| r.vertical().len())
            .cloned()
    }

    pub fn max_overlapping_tile_direct_left(&self) -> Option<Range2D> {
        self.overlapping_tiles_direct_left()
            .iter()
            .max_by_key(|r| r.horisontal().len())
            .cloned()
    }

    pub fn max_overlapping_tile_direct_right(&self) -> Option<Range2D> {
        self.overlapping_tiles_direct_right()
            .iter()
            .max_by_key(|r| r.horisontal().len())
            .cloned()
    }

    pub fn clone_with_image(&self, image: &Array2<u8>) -> Self {
        Tile {
            range2d: self.range2d.clone(),
            image: image.clone(),
            overlapping_tiles_top: self.overlapping_tiles_top.clone(),
            overlapping_tiles_bottom: self.overlapping_tiles_bottom.clone(),
            overlapping_tiles_left: self.overlapping_tiles_left.clone(),
            overlapping_tiles_right: self.overlapping_tiles_right.clone(),
            overlap_ranges_top: self.overlap_ranges_top.clone(),
            overlap_ranges_bottom: self.overlap_ranges_bottom.clone(),
            overlap_ranges_left: self.overlap_ranges_left.clone(),
            overlap_ranges_right: self.overlap_ranges_right.clone(),
        }
    }
}

pub fn plural_s(length: usize) -> String {
    if length == 1 {
        return "".to_string()
    } else {
        return 's'.to_string()
    }
}

pub fn save_array(path: &Path, array: &Array2<u8>) -> Result<(), Error> {
    let image = image::DynamicImage::ImageLuma8(array2_to_image(array));
    image.save(path)?;
    Ok(())
}

pub fn common_path<I, P>(paths: I) -> Option<PathBuf>
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    let mut iter = paths.into_iter();
    let mut ret = iter.next()?.as_ref().to_path_buf();
    for path in iter {
        if let Some(r) = common_path_between_two(ret, path.as_ref()) {
            ret = r;
        } else {
            return None;
        }
    }
    if ret.extension().is_some() {
        // In case input paths has length 1
        ret.parent().map(PathBuf::from)
    } else {
        Some(ret)
    }
}

fn common_path_between_two<A: AsRef<Path>, B: AsRef<Path>>(a: A, b: B) -> Option<PathBuf> {
    let a = a.as_ref().components();
    let b = b.as_ref().components();
    let mut common = PathBuf::new();
    let mut found = false;
    for (one, two) in a.zip(b) {
        if one == two {
            common.push(one);
            found = true;
        } else {
            break;
        }
    }
    if found {
        Some(common)
    } else {
        None
    }
}

fn is_image_path(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .map(|e| IMAGE_EXTENSIONS.contains(&e.to_string_lossy().into_owned().as_str()))
            .unwrap_or(false)
}

pub fn find_image_files(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .map(|e| e.path().to_path_buf())
        .filter(|p| is_image_path(p))
        .collect()
}

pub fn find_image_folders(root: &Path) -> Vec<PathBuf> {
    assert!(
        root.is_dir(),
        "Input to find_image_folders must be a directory"
    );
    let image_filepaths = find_image_files(root);
    let image_folders: Vec<PathBuf> = image_filepaths
        .iter()
        .map(|p| PathBuf::from(p.parent().unwrap()))
        .collect();
    let unique_image_folders: HashSet<PathBuf> = image_folders.iter().cloned().collect();
    unique_image_folders
        .iter()
        .cloned()
        .collect::<Vec<PathBuf>>()
}

fn check_image_path(path: &Path) -> Result<PathBuf, Error> {
    if !path.is_file() {
        return Err(anyhow!("Path does not exist: {}", path.display()));
    }
    if !IMAGE_EXTENSIONS.contains(&path.extension().unwrap().to_str().unwrap()) {
        return Err(anyhow!("Not an image path: {}", path.display()));
    }
    Ok(PathBuf::from(path))
}

pub fn read_text_file(input: &Path) -> Result<Vec<PathBuf>, Error> {
    let file = File::open(input)?;
    let reader = BufReader::new(file);
    let mut image_paths = Vec::<PathBuf>::new();
    for line in reader.lines() {
        let line = line?;
        let image_path = check_image_path(Path::new(&line))?;
        image_paths.push(image_path);
    }
    Ok(image_paths)
}

#[derive(Debug, Deserialize)]
struct Record {
    #[serde(rename = "ImagePath")]
    image_path: String,
    #[serde(rename = "MaskPath")]
    mask_path: Option<String>,
}

impl Record {
    fn image_path(&self) -> String {
        self.image_path.clone()
    }

    fn mask_path(&self) -> Option<String> {
        self.mask_path.clone()
    }
}

pub fn read_csv_file(input: &Path) -> Result<Vec<PathBuf>, Error> {
    let mut reader = csv::Reader::from_path(input)?;
    let mut image_paths = Vec::<PathBuf>::new();
    for result in reader.deserialize() {
        let record: Record = result?;
        let path = record.image_path();
        let path = check_image_path(Path::new(&path))?;
        image_paths.push(path);
        if let Some(path) = record.mask_path() {
            let path = check_image_path(Path::new(&path))?;
            image_paths.push(path);
        }
    }
    Ok(image_paths)
}

pub fn image_to_array2(image: &image::GrayImage) -> Array2<u8> {
    let mut arr = Array2::<u8>::zeros((image.height() as usize, image.width() as usize));
    for (c, r, p) in image.enumerate_pixels() {
        arr[[r as usize, c as usize]] = p[0];
    }
    arr
}

pub fn array2_to_image(array: &Array2<u8>) -> image::GrayImage {
    let mut image = image::GrayImage::new(array.ncols() as u32, array.nrows() as u32);
    for ((r, c), &v) in array.indexed_iter() {
        image.put_pixel(c as u32, r as u32, image::Luma([v]));
    }
    image
}

pub fn is_overlapping(r1: &Range<usize>, r2: &Range<usize>) -> bool {
    r1.contains(&r2.start) || r2.contains(&r1.start)
}

pub fn tiles_overlap(this: &Range2D, other: &Range2D) -> bool {
    is_overlapping(&this.vertical(), &other.vertical())
        && is_overlapping(&this.horisontal(), &other.horisontal())
}

/// Overlap above
/// ```text,ignore
///                Other top -> OOOOOOOOOOOO
///                             O          O
///  Coord                      O Other    O
///    |                        O          O
///    |            This top -> O   TTTTTTTØTTTT  |
///    |                        O   T      O   T  | Overlap
///    |        Other bottom -> OOOOØOOOOOOO   T  |
///    V                            T          T
///                                 T This     T
///                                 T          T
///              This bottom ->     TTTTTTTTTTTT
/// ```
pub fn overlap_top(this: &Range2D, other: &Range2D) -> Option<Range<usize>> {
    if tiles_overlap(this, other) {
        if this.top() > other.top() && this.top() < other.bottom() && this.bottom() > other.bottom()
        {
            Some(this.top()..other.bottom())
        } else {
            None
        }
    } else {
        None
    }
}

/// Overlap below
/// ```text,ignore
///                 This top -> TTTTTTTTTTTT
///                             T          T
///  Coord                      T Other    T
///    |                        T          T
///    |           Other top -> T   OOOOOOOØOOOO |
///    |                        T   O      T   O | Overlap
///    |         This bottom -> TTTTØTTTTTTT   O |
///    V                            O          O
///                                 O Other    O
///                                 O          O
///             Other bottom ->     OOOOOOOOOOOO
/// ```
pub fn overlap_bottom(this: &Range2D, other: &Range2D) -> Option<Range<usize>> {
    if tiles_overlap(this, other) {
        if other.top() > this.top() && other.top() < this.bottom() && other.bottom() > this.bottom()
        {
            Some(other.top()..this.bottom())
        } else {
            None
        }
    } else {
        None
    }
}

/// Overlap to the left
/// ```text,ignore
///     Coord -------------------->
///
///     Other          This     Other        This
///     left           left     right        right
///     V              V        V            V
///     OOOOOOOOOOOOOOOOOOOOOOOOO
///     O                       O
///     O    Other     TTTTTTTTTØTTTTTTTTTTTTT
///     O              T        O            T
///     OOOOOOOOOOOOOOOØOOOOOOOOO  This      T
///                    T                     T
///                    TTTTTTTTTTTTTTTTTTTTTTT
///                    ----------
///                    Overlap
///
pub fn overlap_left(this: &Range2D, other: &Range2D) -> Option<Range<usize>> {
    if tiles_overlap(this, other) {
        if this.left() > other.left() && this.left() < other.right() && this.right() > other.right()
        {
            Some(this.left()..other.right())
        } else {
            None
        }
    } else {
        None
    }
}

/// Overlap to the right
/// ```text,ignore
///     Coord -------------------->
///
///     This           Other    This         Other
///     left           left     right        right
///     V              V        V            V
///     TTTTTTTTTTTTTTTTTTTTTTTTT
///     T                       T
///     T    This      OOOOOOOOOØOOOOOOOOOOOOO
///     T              O        T            O
///     TTTTTTTTTTTTTTTØTTTTTTTTT  This      O
///                    O                     O
///                    OOOOOOOOOOOOOOOOOOOOOOO
///                    ----------
///                    Overlap
///
/// ```
pub fn overlap_right(this: &Range2D, other: &Range2D) -> Option<Range<usize>> {
    if tiles_overlap(this, other) {
        if other.left() > this.left() && other.left() < this.right() && other.right() > this.right()
        {
            Some(other.left()..this.right())
        } else {
            None
        }
    } else {
        None
    }
}

/// Test asset
// TODO: Replace with function from float_cmp crate
#[allow(unused)]
pub fn compare_float(n1: f32, n2: f32) -> bool {
    (n1 - n2).abs() < 1.0e-6
}

/// Test asset
#[allow(unused)]
pub fn compare_arrays(result: &Array2<f32>, expected: &Array2<f32>, name: &'static str) {
    assert_eq!(result.dim(), expected.dim(), "Unequal array dimension");
    let mut ok = true;
    for ((i, j), r) in result.indexed_iter() {
        let e = expected[[i, j]];
        if !compare_float(*r, e) {
            ok = false;
            println!("At ({}, {}): result {} vs expected {}", i, j, r, e);
        }
    }
    assert!(ok, "{}", name);
}

/// Test asset
#[allow(unused)]
pub fn test_image_exact() -> Array2<u8> {
    arr2(&[
        [10, 10, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 10, 10, 10],
        [10, 10, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 10, 10, 10],
        [10, 10, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 10],
        [10, 10, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 10],
        [10, 10, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 10],
        [10, 10, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 10],
        [10, 10, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 10],
        [10, 10, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 10, 10, 10],
        [10, 10, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 10, 10, 10],
        [10, 10, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 10, 10, 10],
    ])
}

/// Test asset
#[allow(unused)]
pub fn test_image_round() -> Array2<u8> {
    arr2(&[
        [10, 10, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 10, 10, 10],
        [10, 10, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 10, 10, 10],
        [10, 10, 50, 52, 52, 52, 52, 50, 52, 52, 52, 50, 10, 10, 10],
        [10, 10, 50, 52, 52, 52, 52, 50, 52, 52, 52, 50, 10, 10, 10],
        [10, 10, 50, 52, 52, 52, 52, 50, 52, 52, 52, 50, 10, 10, 10],
        [10, 10, 50, 52, 52, 52, 52, 50, 52, 52, 52, 50, 10, 10, 10],
        [10, 10, 50, 52, 52, 52, 52, 50, 52, 52, 52, 50, 10, 10, 10],
        [10, 10, 90, 92, 92, 92, 92, 90, 92, 92, 92, 90, 10, 10, 10],
        [10, 10, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 10, 10, 10],
        [10, 10, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 10, 10, 10],
    ])
}

/// Test asset
#[allow(unused)]
pub fn test_tile_11() -> (Range2D, Array2<u8>) {
    let im = arr2(&[
        [10, 10, 90, 90, 90, 90, 90, 90],
        [10, 10, 90, 90, 90, 90, 90, 90],
        [10, 10, 50, 50, 50, 50, 50, 50],
        [10, 10, 50, 50, 50, 50, 50, 50],
        [10, 10, 50, 50, 50, 50, 50, 50],
        [10, 10, 50, 50, 50, 50, 50, 50],
        [10, 10, 50, 50, 50, 50, 50, 50],
        [10, 10, 90, 90, 90, 90, 90, 90],
    ]);
    let ver = 0..8;
    let hor = 0..8;
    (Range2D::new(ver, hor), im)
}

/// Test asset
#[allow(unused)]
pub fn test_tile_12() -> (Range2D, Array2<u8>) {
    let im = arr2(&[
        [90, 90, 90, 90, 90, 90, 90, 90],
        [90, 90, 90, 90, 90, 90, 90, 90],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [90, 90, 90, 90, 90, 90, 90, 90],
    ]);
    let ver = 0..8;
    let hor = 3..11;
    (Range2D::new(ver, hor), im)
}

/// Test asset
#[allow(unused)]
pub fn test_tile_13() -> (Range2D, Array2<u8>) {
    let im = arr2(&[
        [90, 90, 90, 90, 90, 10, 10, 10],
        [90, 90, 90, 90, 90, 10, 10, 10],
        [50, 50, 50, 50, 50, 10, 10, 10],
        [50, 50, 50, 50, 50, 10, 10, 10],
        [50, 50, 50, 50, 50, 10, 10, 10],
        [50, 50, 50, 50, 50, 10, 10, 10],
        [50, 50, 50, 50, 50, 10, 10, 10],
        [90, 90, 90, 90, 90, 10, 10, 10],
    ]);
    let ver = 0..8;
    let hor = 7..15;
    (Range2D::new(ver, hor), im)
}

/// Test asset
#[allow(unused)]
pub fn test_tile_21() -> (Range2D, Array2<u8>) {
    let im = arr2(&[
        [10, 10, 50, 50, 50, 50, 50, 50],
        [10, 10, 50, 50, 50, 50, 50, 50],
        [10, 10, 50, 50, 50, 50, 50, 50],
        [10, 10, 50, 50, 50, 50, 50, 50],
        [10, 10, 50, 50, 50, 50, 50, 50],
        [10, 10, 90, 90, 90, 90, 90, 90],
        [10, 10, 90, 90, 90, 90, 90, 90],
        [10, 10, 90, 90, 90, 90, 90, 90],
    ]);
    let ver = 2..10;
    let hor = 0..8;
    (Range2D::new(ver, hor), im)
}

/// Test asset
#[allow(unused)]
pub fn test_tile_22() -> (Range2D, Array2<u8>) {
    let im = arr2(&[
        [50, 50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [90, 90, 90, 90, 90, 90, 90, 90],
        [90, 90, 90, 90, 90, 90, 90, 90],
        [90, 90, 90, 90, 90, 90, 90, 90],
    ]);
    let ver = 2..10;
    let hor = 3..11;
    (Range2D::new(ver, hor), im)
}

/// Test asset
#[allow(unused)]
pub fn test_tile_23() -> (Range2D, Array2<u8>) {
    let im = arr2(&[
        [50, 50, 50, 50, 50, 10, 10, 10],
        [50, 50, 50, 50, 50, 10, 10, 10],
        [50, 50, 50, 50, 50, 10, 10, 10],
        [50, 50, 50, 50, 50, 10, 10, 10],
        [50, 50, 50, 50, 50, 10, 10, 10],
        [90, 90, 90, 90, 90, 10, 10, 10],
        [90, 90, 90, 90, 90, 10, 10, 10],
        [90, 90, 90, 90, 90, 10, 10, 10],
    ]);
    let ver = 2..10;
    let hor = 7..15;
    (Range2D::new(ver, hor), im)
}
