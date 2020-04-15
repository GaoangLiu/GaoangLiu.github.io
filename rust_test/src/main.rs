use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;

fn read_file(file_name: String) -> std::io::Result<String> {
    let mut file = File::open(file_name)?;
    let mut buf_reader = BufReader::new(file);
    let mut contents = String::new();
    buf_reader.read_to_string(&mut contents)?;
    Ok(contents)
}

fn read_file_2(file_name: String) -> String{
    let mut file = File::open(file_name).expect("Unable to open");
    let mut buf_reader = BufReader::new(file);
    let mut contents = String::new();
    buf_reader.read_to_string(&mut contents).unwrap();
    contents
}

fn file_exist(file_name: String) -> std::io::Result<()> {
    let mut f = File::open(file_name)?;
    println!("{:?}", f);
    Ok(())
}

fn create_file(file_name: String) -> std::io::Result<()>{
    let mut f = File::create(file_name)?;
    Ok(())
}

fn write_to_file(fname: &str) -> std::io::Result<()> {
    let mut file = File::create(fname)?;
    file.write_all(b"Hello, world!")?;
    Ok(())
}

pub fn tes(fname: &str) {
    let mut f = File::create(fname).unwrap();
    f.write_all(b"Hello, world!\n").unwrap();
}

fn main() {
    let ct = read_file_2("src/mains.rs".to_string());
    // match ct {
    //     Ok(s) => println!("{:?}", s), 
    //     Err(err) => eprintln!("EEeeeee {}", err),
    // }
    // println!("{:?}", ct);


    // match file_exist("foo.txt".to_string()) {
    //     Ok(uptime) => println!("uptime: {} seconds", 3),
    //     Err(err) => eprintln!("EEEeeeeeerror: {}", err),
    // };
}