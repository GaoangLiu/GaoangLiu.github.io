use rand::Rng; 

fn main() {
    println!("Hello, world!");
    let x = rand::random::<f32>();
    println!("{}", x);

    let mut rnd = rand::thread_rng();
    println!("{:?}", rnd.gen_range(0, 10));
}