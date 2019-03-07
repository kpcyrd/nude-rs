#[macro_use]
extern crate log;

use env_logger;
use image;
use image::GenericImageView;
use std::env;

fn main() {
    env_logger::init();

    let path = env::args().skip(1).next().expect("argv[1]");

    let img = image::open(path).expect("failed to open");
    info!("dimensions {:?}", img.dimensions());

    let nude = nude::scan(&img).analyse();
    println!("nude={:?}", nude);
}
