extern crate rand;
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};

pub struct Helper;
impl Helper {
    pub fn stringify(str_vector: Vec<&str>) -> Vec<String> {
        // Convert a vector of &str to vector or String for coding convenience
        str_vector.iter().map(|c| c.to_string()).collect::<Vec<String>>()
    }
}

fn main() {
    let data = vec![vec![2, 3, 4], vec![9, 8, 1]];
    let d_flatten:Vec<i32> = data.into_iter().flatten().collect();
    println!("{}", d_flatten.into_iter().sum::<i32>());
}
