use std::env;
use std::time::Instant;


fn main() {
    env_logger::init();

    let imgs = env::args().skip(1)
        .map(image::open)
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to load images");

    let start = Instant::now();

    for img in imgs {
        for _ in 0..10 {
            let _ = nude::scan(&img).analyse();
        }
    }

    let elapsed = start.elapsed();

    println!("Execution time: {}ms ({}s)", elapsed.as_millis(), elapsed.as_secs());

}
