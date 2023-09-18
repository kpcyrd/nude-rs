# nude-rs [![Build Status][travis-img]][travis] [![crates.io][crates-img]][crates] [![docs.rs][docs-img]][docs]

[travis-img]:   https://travis-ci.com/kpcyrd/nude-rs.svg?branch=master
[travis]:       https://travis-ci.com/kpcyrd/nude-rs
[crates-img]:   https://img.shields.io/crates/v/nude.svg
[crates]:       https://crates.io/crates/nude
[docs-img]:     https://docs.rs/nude/badge.svg
[docs]:         https://docs.rs/nude

High performance nudity detection in rust. This is a port based on [nude.js]
and [nude.py].

[nude.js]: https://github.com/pa7/nude.js
[nude.py]: https://github.com/hhatto/nude.py

![benchmark](docs/benchmark.png)

We are currently going for an identical nudity detection algorithm, future
versions may tweak parameters but due to the complexity of this topic we can't
provide actual support for false positives or negatives. You are welcome to
discuss ideas in the issue tracker but you may have to write a patch yourself.

The original implementation is based on [this paper][0] (An Algorithm for Nudity Detection).

[0]: https://web.archive.org/web/20190415181208/https://be82ac79-a-aff4c786-s-sites.googlegroups.com/a/dcs.upd.edu.ph/csp-proceedings/Home/pcsc-2005/AI4.pdf?attachauth=ANoY7cqxudROUdZ3sxZE_dDeY3KTWsN11uK2Mmckptf80SlcKmzvqApC2yKu-snwCKqJSpmTRbO0-t-JhEzcKqPLy6NiHci5d1OwViFCm2yRIaNm2rcvASaH4neMXimlDis8hvW7VYZGWZ-feJyeu_cl-qeMScVOsVMET6yCRyjYT10nBgxmaa9SQXBke8rD0nHjlB4VgmpBbFd1ALxboziXyk6gNl70hqF6qho_ro3IqquNvFBEay4%3D&attredirects=0

Please consider this library experimental.

## Usage as a library

```rust
extern crate image;
extern crate nude;

let img = image::open("test_data/test2.jpg").expect("failed to open");
let nudity = nude::scan(&img).analyse();
println!("nudity={:?}", nudity);
```

## Running the example binary
```
cargo run --release --example scan test_data/test2.jpg
```

## License
This project is free software released under the LGPL3+ license.
