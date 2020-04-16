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
    let rand_string: String = thread_rng().sample_iter(&Alphanumeric).take(30).collect();

    println!("{}", rand_string);

    let s: String = "Sex more.".into();
    println!("{:?}", s);

    // let vs = vec!["a".to_owned(), "b".into()];
    let vs = Helper::stringify(vec!["a", "b", "c"]);
    println!("{:?}", vs);

    let v = vec![1_i32, 9, 8];
    v[99];
}
