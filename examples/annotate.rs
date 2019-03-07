#[macro_use]
extern crate log;

use env_logger;
use image;
use image::GenericImageView;
use std::env;

fn main() {
    env_logger::init();

    let path = env::args().skip(1).next().expect("argv[1]");

    let mut img = image::open(path).expect("failed to open");
    info!("dimensions {:?}", img.dimensions());

    nude::scan(&img).colorize_regions(&mut img);
    img.save("output.jpg").expect("failed to save");
}
