//! ```rust
//! extern crate image;
//! extern crate nude;
//!
//! let img = image::open("test_data/test2.jpg").expect("failed to open");
//! let nudity = nude::scan(&img).analyse();
//! println!("nudity={:?}", nudity);
//! ```

use image::DynamicImage;
use image::GenericImage;
use image::GenericImageView;
// Re-enable this after rust 1.33.0 is more common
// https://github.com/kpcyrd/sn0int/issues/89
// use image::Pixel as _;
use image::Pixel as ImagePixel;
use image::Primitive;
use image::Rgb;
use image::Rgba;
use log::*;
use rand::Rng;

/// A classified pixel in an image
#[derive(Debug, Clone)]
pub struct Pixel {
    id: u32,
    skin: bool,
    region: u64,
    x: u32,
    y: u32,
    checked: bool,
}

impl Pixel {
    pub fn new(skin: bool, x: u32, y: u32) -> Pixel {
        let id = (y * x) + x;
        Pixel {
            id,
            skin,
            region: 0,
            x,
            y,
            checked: false,
        }
    }
}

/// The classification of an image
pub struct Scan {
    skin_regions: Vec<Vec<Pixel>>,
    width: u32,
    height: u32,
}

/// Scan an image skin regions
pub fn scan(image: &DynamicImage) -> Scan {
    let (width, height) = image.dimensions();

    let mut skin_map = Vec::new();

    let mut detected_regions: Vec<Vec<Pixel>> = Vec::new();
    let mut merge_regions: Vec<Vec<u64>> = Vec::new();

    let mut last_from = None;
    let mut last_to = None;

    for (x, y, pixel) in image.pixels() {
        let rgb = pixel.to_rgb();

        let u = (x + y * width) as i32 + 1;

        if classify_skin(rgb) {
            trace!("detected skin at x={}, y={}", x, y);
            skin_map.push(Pixel::new(true, x, y));

            let mut region = None;
            let mut checker = false;

            for index in &[
                u - 2,
                u - (width as i32) - 2,
                u - (width as i32) - 1,
                u - (width as i32),
            ] {
                if *index < 0 {
                    continue;
                }

                let index = *index as usize;
                if let Some(pixel) = skin_map.get(index) {
                    if pixel.skin {
                        if let Some(region) = region {
                            if pixel.region != region
                                && last_from != Some(region)
                                && last_to != Some(pixel.region)
                            {
                                last_from = Some(region);
                                last_to = Some(pixel.region);
                                add_merge(&mut merge_regions, region, pixel.region);
                            }
                        }
                        region = Some(pixel.region);
                        checker = true;
                    }
                }
            }

            if !checker {
                let last = skin_map.last_mut().unwrap();
                last.region = detected_regions.len() as u64;
                detected_regions.push(vec![last.clone()]);
                continue;
            } else {
                if let Some(region) = region {
                    if detected_regions.len() - 1 < region as usize {
                        detected_regions.push(Vec::new());
                    }

                    let last = skin_map.last_mut().unwrap();
                    last.region = region;
                    detected_regions[region as usize].push(last.clone());
                }
            }
        } else {
            trace!("no skin at x={}, y={}", x, y);
            skin_map.push(Pixel::new(false, x, y));
        }
    }

    let skin_regions = merge(detected_regions, merge_regions);
    Scan {
        skin_regions,
        width,
        height,
    }
}

fn add_merge(merge_regions: &mut Vec<Vec<u64>>, from: u64, to: u64) {
    let mut from_index = None;
    let mut to_index = None;

    for (index, region) in merge_regions.iter().enumerate() {
        for r_index in region {
            if *r_index == from {
                from_index = Some(index);
            }
            if *r_index == to {
                to_index = Some(index);
            }
        }
    }

    match (from_index, to_index) {
        (Some(from_index), Some(to_index)) if from_index == to_index => {
            // skip
        }
        (None, None) => {
            merge_regions.push(vec![from, to]);
        }
        (Some(from_index), None) => {
            merge_regions[from_index].push(to);
        }
        (None, Some(to_index)) => {
            merge_regions[to_index].push(from);
        }
        (Some(mut from_index), Some(to_index)) => {
            // .remove(to_index casues a shift. if that affects our index, adjust
            if to_index < from_index {
                from_index -= 1;
            }
            let mut x = merge_regions.remove(to_index);
            merge_regions[from_index].append(&mut x);
        }
    }
}

fn merge(mut detected_regions: Vec<Vec<Pixel>>, merge_regions: Vec<Vec<u64>>) -> Vec<Vec<Pixel>> {
    let mut new_detected_regions = Vec::new();

    // merging detected regions
    for (index, region) in merge_regions.iter().enumerate() {
        new_detected_regions.push(Vec::new());

        for r_index in region {
            new_detected_regions[index].append(&mut detected_regions[*r_index as usize]);
        }
    }

    // push the rest of the regions to the new_detected_regions array
    // (regions without merging)
    for region in detected_regions {
        if region.len() > 0 {
            new_detected_regions.push(region)
        }
    }

    // clean up
    clear_regions(new_detected_regions)
}

fn clear_regions(detected_regions: Vec<Vec<Pixel>>) -> Vec<Vec<Pixel>> {
    let mut skin_regions = Vec::new();
    for det in detected_regions {
        if det.len() > 30 {
            skin_regions.push(det)
        }
    }
    skin_regions
}

impl Scan {
    // fn analyse_regions(&mut self, mut skin_regions: Vec<Vec<Pixel>>, width: u32, height: u32) -> bool {
    pub fn analyse(&mut self) -> Analysis {
        let skin_regions = &mut self.skin_regions;
        let total_pixels = self.width * self.height;

        // count total skin pixels
        let total_skin: usize = skin_regions.iter().map(|x| x.len()).sum();

        // check if there are more than 15% skin pixel in the image
        let skin_percent = 100.0 / f64::from(total_pixels) * f64::from(total_skin as u32);
        debug!("total skin percent is {}%", skin_percent);

        // if there are less than 3 regions
        debug!("skin regions: {}", skin_regions.len());
        if skin_regions.len() < 3 {
            debug!("not nude - less than 3 regions");
            return Analysis {
                nude: false,
                skin_percent,
            };
        }

        // sort the detected regions by size
        debug!("sorting skin regions");
        skin_regions.sort_by(|a, b| a.len().cmp(&b.len()).reverse());

        if skin_percent < 15.0 {
            // if the percentage lower than 15, it's not nude!
            debug!("not nude - skin percent is < 15%");
            return Analysis {
                nude: false,
                skin_percent,
            };
        }

        // check if the largest skin region is less than 35% of the total skin count
        // AND if the second largest region is less than 30% of the total skin count
        // AND if the third largest region is less than 30% of the total skin count
        if total_skin_percent(&skin_regions[0], total_skin) < 35.0
            && total_skin_percent(&skin_regions[1], total_skin) < 30.0
            && total_skin_percent(&skin_regions[2], total_skin) < 30.0
        {
            // the image is not nude.
            debug!(
                "not nude - less than 35%,30%,30% in the biggest areas: {}%, {}%, {}%",
                total_skin_percent(&skin_regions[0], total_skin),
                total_skin_percent(&skin_regions[1], total_skin),
                total_skin_percent(&skin_regions[2], total_skin)
            );
            return Analysis {
                nude: false,
                skin_percent,
            };
        }

        // check if the number of skin pixels in the largest region is less than 45% of the total skin count
        if total_skin_percent(&skin_regions[0], total_skin) < 45.0 {
            // it's not nude
            debug!(
                "not nude - the biggest region contains less than 45%: {}%",
                total_skin_percent(&skin_regions[0], total_skin)
            );
            return Analysis {
                nude: false,
                skin_percent,
            };
        }

        // TODO:
        // build the bounding polygon by the regions edge values:
        // Identify the leftmost, the uppermost, the rightmost, and the lowermost skin pixels of the three largest skin regions.
        // Use these points as the corner points of a bounding polygon.

        // TODO:
        // check if the total skin count is less than 30% of the total number of pixels
        // AND the number of skin pixels within the bounding polygon is less than 55% of the size of the polygon
        // if this condition is true, it's not nude.

        // TODO: include bounding polygon functionality
        // if there are more than 60 skin regions and the average intensity within the polygon is less than 0.25
        // the image is not nude
        if skin_regions.len() > 60 {
            debug!("not nude - more than 60 skin regions");
            return Analysis {
                nude: false,
                skin_percent,
            };
        }

        // otherwise it is nude
        return Analysis {
            nude: true,
            skin_percent,
        };
    }

    #[inline]
    pub fn is_nude(&mut self) -> bool {
        let analysis = self.analyse();
        analysis.nude
    }

    pub fn colorize_regions(&self, img: &mut DynamicImage) {
        // TODO: check img has same size

        let mut rng = rand::thread_rng();
        for region in &self.skin_regions {
            let r: u8 = rng.gen();
            let g: u8 = rng.gen();
            let b: u8 = rng.gen();

            for px in region {
                img.put_pixel(px.x, px.y, Rgba { data: [r, g, b, 0] });
            }
        }
    }
}

/// The final analysis of an image
#[derive(Debug, Clone, PartialEq)]
pub struct Analysis {
    /// The classification whether this image is a nudie
    pub nude: bool,
    /// The percentage of skin pixels in the image
    pub skin_percent: f64,
}

impl Analysis {
    /// Returns a combined score of `nude` and `skin_percent`.
    /// score > 1.0 indicates a detected nudie.
    ///
    /// ```
    /// # use nude::Analysis;
    /// let analysis = Analysis {
    ///     nude: true,
    ///     skin_percent: 65.34,
    /// };
    /// assert_eq!(analysis.score(), 1.6534);
    ///
    /// let analysis = Analysis {
    ///     nude: false,
    ///     skin_percent: 12.3,
    /// };
    /// assert_eq!(analysis.score(), 0.12300000000000001);
    /// ```
    #[inline]
    pub fn score(&self) -> f64 {
        let nude = if self.nude { 1.0 } else { 0.0 };
        nude + (self.skin_percent / 100.0)
    }
}

#[inline]
fn total_skin_percent(skin_region: &[Pixel], total_skin: usize) -> f64 {
    100.0 / f64::from(total_skin as u32) * f64::from(skin_region.len() as u32)
}

#[inline]
fn math_max(r: f64, g: f64, b: f64) -> f64 {
    let mut x = r;
    if g > x {
        x = g;
    }
    if b > x {
        x = b;
    }
    x
}

#[inline]
fn math_min(r: f64, g: f64, b: f64) -> f64 {
    let mut x = r;
    if g < x {
        x = g;
    }
    if b < x {
        x = b;
    }
    x
}

/// Determine if the pixel is likely to be a skin pixel
pub fn classify_skin<T: Primitive>(rgb: Rgb<T>) -> bool
where
    T: Into<f64>,
{
    let r: f64 = rgb[0].into();
    let g: f64 = rgb[1].into();
    let b: f64 = rgb[2].into();

    // A Survey on Pixel-Based Skin Color Detection Techniques
    let rgb_classifier = (r > 95.0)
        && (g > 40.0 && g < 100.0)
        && (b > 20.0)
        && ((math_max(r, g, b) - math_min(r, g, b)) > 15.0)
        && ((r - g).abs() > 15.0)
        && (r > g)
        && (r > b);

    let nurgb = to_normalized_rgb(rgb);
    let nr = nurgb.0;
    let ng = nurgb.1;
    let _nb = nurgb.2;
    let norm_rgb_classifier = ((nr / ng) > 1.185)
        && (((r * b) / (r + g + b).powi(2)) > 0.107)
        && (((r * g) / (r + g + b).powi(2)) > 0.112);

    //hsv = toHsv(r, g, b),
    //h = hsv[0]*100,
    //s = hsv[1],
    //hsv_classifier = (h < 50 && h > 0 && s > 0.23 && s < 0.68);

    let hsv = to_hsv_test(rgb);
    let h = hsv.0;
    let s = hsv.1;
    let hsv_classifier = h > 0.0 && h < 35.0 && s > 0.23 && s < 0.68;

    /*
     * ycc doesnt work

    ycc = toYcc(r, g, b),
    y = ycc[0],
    cb = ycc[1],
    cr = ycc[2],
    yccClassifier = ((y > 80) && (cb > 77 && cb < 127) && (cr > 133 && cr < 173));
    */

    rgb_classifier || norm_rgb_classifier || hsv_classifier
}

/*
fn to_ycc<T: Primitive>(rgb: Rgb<T>) -> (f64, f64, f64)
    where T: Into<f64>
{
    let r: f64 = rgb[0].into() / 255.0;
    let g: f64 = rgb[1].into() / 255.0;
    let b: f64 = rgb[2].into() / 255.0;

    let y = 0.299*r + 0.587*g + 0.114*b;

    let cr = r - y;
    let cb = b - y;

    (y, cr, cb)
}

fn to_hsv<T: Primitive>(rgb: Rgb<T>) -> (f64, f64, f64)
    where
        T: Into<f64>,
{
    let r: f64 = rgb[0].into();
    let g: f64 = rgb[1].into();
    let b: f64 = rgb[2].into();

    /*
    let hue = Math.acos((0.5*((r-g)+(r-b)))/(Math.sqrt((Math.pow((r-g),2)+((r-b)*(g-b))))));
    */
/*
let hue = Math.acos(
    (0.5*((r-g)+(r-b)))
    /
    (Math.sqrt(
        (
            Math.pow((r-g),2)
            +
            ((r-b)*(g-b))
        )
    ))
);
*/
let hue = ((0.5*((r-g)+(r-b)))/((r-g).powi(2)+((r-b)*(g-b))).sqrt()).acos();
let saturation = 1.0-(3.0*(math_min(r,g,b)/(r+g+b)));
let value = (1.0/3.0)*(r+g+b);

(hue, saturation, value)
}
*/

fn to_hsv_test<T: Primitive>(rgb: Rgb<T>) -> (f64, f64, f64)
where
    T: Into<f64>,
{
    let r: f64 = rgb[0].into();
    let g: f64 = rgb[1].into();
    let b: f64 = rgb[2].into();

    let mx = math_max(r, g, b);
    let mn = math_min(r, g, b);
    let dif = mx - mn;

    let mut h = if mx == r {
        (g - b) / dif
    } else if mx == g {
        2.0 + ((g - r) / dif)
    } else {
        4.0 + ((r - g) / dif)
    };

    h *= 60.0;

    if h < 0.0 {
        h += 360.0;
    }

    (
        h,
        1.0 - (3.0 * (mn / (r + g + b))),
        (1.0 / 3.0) * (r + g + b),
    )
}

fn to_normalized_rgb<T: Primitive>(rgb: Rgb<T>) -> (f64, f64, f64)
where
    T: Into<f64>,
{
    let r: f64 = rgb[0].into();
    let g: f64 = rgb[1].into();
    let b: f64 = rgb[2].into();

    let sum = r + g + b;

    (r / sum, g / sum, b / sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_skin() {
        let skin = classify_skin(Rgb {
            data: [219, 191, 177],
        });
        assert!(!skin);

        let skin = classify_skin(Rgb {
            data: [223, 199, 187],
        });
        assert!(!skin);

        let skin = classify_skin(Rgb {
            data: [112, 110, 89],
        });
        assert!(!skin);

        let skin = classify_skin(Rgb {
            data: [175, 125, 102],
        });
        assert!(skin);

        let skin = classify_skin(Rgb {
            data: [127, 83, 58],
        });
        assert!(skin);
    }

    /*
    #[test]
    fn test_to_ycc() {
        let ycc = to_ycc(Rgb {
            data: [50.0, 100.0, 200.0],
        });

        println!("y={}, cr={}, cb={}", ycc.0, ycc.1, ycc.2);
        assert_eq!(ycc, (0.37823529411764706, -0.18215686274509804, 0.406078431372549));
    }

    #[test]
    fn test_to_hsv() {
        let hsv = to_hsv(Rgb {
            data: [50, 100, 200],
        });

        println!("hue={}, saturation={}, value={}", hsv.0, hsv.1, hsv.2);
        // nude.js result: assert_eq!(hsv, (2.4278682746450277, 0.5714285714285714, 116.66666666666666));
        assert_eq!(hsv, (2.4278682746450273, 0.5714285714285714, 116.66666666666666));
    }
    */

    #[test]
    fn test_to_hsv_test() {
        let hsv = to_hsv_test(Rgb {
            data: [50, 100, 200],
        });

        println!("hue={}, saturation={}, value={}", hsv.0, hsv.1, hsv.2);
        assert_eq!(hsv, (220.0, 0.5714285714285714, 116.66666666666666));
    }

    #[test]
    fn test_to_normalized_rgb() {
        let rgb = to_normalized_rgb(Rgb {
            data: [50.0, 100.0, 200.0],
        });

        println!("r={}, g={}, b={}", rgb.0, rgb.1, rgb.2);
        assert_eq!(
            rgb,
            (0.14285714285714285, 0.2857142857142857, 0.5714285714285714)
        );
    }

    #[test]
    fn integration_damita() {
        let img = image::open("test_data/damita.jpg").expect("failed to open");
        let score = scan(&img).analyse().score();
        assert_eq!(score, 0.07692266666666667);
    }

    #[test]
    fn integration_damita2() {
        let img = image::open("test_data/damita2.jpg").expect("failed to open");
        let score = scan(&img).analyse().score();
        assert_eq!(score, 0.10047466666666667);
    }

    #[test]
    fn integration_test2() {
        let img = image::open("test_data/test2.jpg").expect("failed to open");
        let score = scan(&img).analyse().score();
        assert_eq!(score, 1.3948800000000001);
    }

    #[test]
    fn integration_test6() {
        let img = image::open("test_data/test6.jpg").expect("failed to open");
        let score = scan(&img).analyse().score();
        assert_eq!(score, 0.31169066666666667);
    }
}
