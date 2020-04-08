fn print_sum(v: Vec<i32>) {
    println!("{}", v[0] + v[1]);
    // v is dropped and deallocated here
}

fn myprint(v: Vec<i32>) {
    println!("{:?}", v);
}

fn printi32(i: i32) {
    println!("{}", i);
}


use std::collections::HashSet; 
use std::any::type_name;

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn main() {
    let mut s = vec![0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
    let i = s.binary_search(&1).unwrap_or_else(|x| x);
    
    print_type_of(&i);
    println!("{:?}", i);
    
    let t: &[i32] = &[];
    let nlast = t.last().unwrap_or_else(|x|);
    print_type_of(&nlast);
    // println!("{}", nlast);
}
