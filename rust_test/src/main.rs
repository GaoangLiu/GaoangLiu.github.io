use std::collections::BinaryHeap; 

fn main() {
    println!("Hello, world!");

    let mut bh = BinaryHeap::new(); 
    bh.push((3, 2));
    bh.push((1, 4));
    bh.push((7, 4));
    let p = bh.peek(); 
    println!("{:?}", p);
    println!("{:?}", bh.len());

    print!("{:?}", bh.into_vec());

    // let a1 = [1_i32, 3, 5, 7, 9, 11]; 
    // let a2 = [2_i32, 4, 6];
    // let mut zipped = a1.iter().zip(a2.iter());
    // println!("{:?}", zipped);
    // // println!("{:?}", zipped.next());
    // for z in zipped {
    //     println!("{:?}", z);
    // }

    // let v = [1_i32, 3, 3, 1987, 3, 3, 5, 7, 3];

    // let idx = v.iter().rposition(|&x| x == 3).unwrap();
    // println!("{}", idx);

    // let vmax = *v.iter().max().unwrap();
    // if vmax == 1987 {
    //     println!("Haha, the type if i32");
    // }
    // println!("Max element {}", vmax);


    
    // let a = [(1_i32, 3_i32), (-1029, 99), (10, 2)];
    // let xa = *a.iter().max_by_key(|x| x.1).unwrap(); 
    // println!("Max by key {:?}", xa);
    
    // let b = [1_i32, 2, 3];
    // let xb = *b.iter().max_by(|x, y| x.cmp(&&(7 + **y))).unwrap();
    // println!("{}", xb);
}
