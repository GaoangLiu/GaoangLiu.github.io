fn print_sum(v: Vec<i32>) {
    println!("{}", v[0] + v[1]);
    // v is dropped and deallocated here
}

fn myprint(v:Vec<i32>){
    println!("{:?}", v);
}

fn main() {
    let mut v = Vec::new(); // creating the resource
    for i in 1..10 {
        v.push(i);
    }
    // at this point, v is using
    // no less than 4000 bytes of memory
    // -------------------
    // transfer ownership to print_sum:
    // print_sum(v);
    // print_sum(v);
    println!("{:?}", v);
    println!("{:?}", v);
    myprint(v);
    // we no longer own nor anyhow control v
    // it would be a compile-time error to try to access v here
    println!("We're done");
    // no deallocation happening here,
    // because print_sum is responsible for everything


    use std::collections::HashMap;
    let mut hm = HashMap::new(); 
    hm.insert("a".to_string(), 3);
    hm.insert("a".to_string(), 3);
    let v = hm.entry("b".to_string()).or_insert(4);
    println!("{}", v);

    println!("{:?}", hm);
}