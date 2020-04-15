extern crate rand;

use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};

fn main() {
    let rand_string: String = thread_rng()
        .sample_iter(&Alphanumeric)
        .take(30)
        .collect();

    println!("{}", rand_string);
}
