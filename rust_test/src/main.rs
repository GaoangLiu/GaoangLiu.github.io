extern crate rand;
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};

use std::collections::BTreeMap;

pub struct Helper;
impl Helper {
    pub fn stringify(str_vector: Vec<&str>) -> Vec<String> {
        // Convert a vector of &str to vector or String for coding convenience
        str_vector.iter().map(|c| c.to_string()).collect::<Vec<String>>()
    }
}

fn main() {
    let ref _is_a_reference = 3;
    _is_a_reference = 4;
    println!("{:?}", _is_a_reference);
}