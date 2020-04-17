// comment?

use std::{i8,i16,i32,i64,f32,f64};

fn main() {
    println!("Somethin'");

    // there is auto kind of functionality in rust
    let x = 10;

    // also variables are immutable, unless
    
    let _y: i16 = 40;
    let y = 20;

    let abc: bool = true;

    let c: char = 'c';
    
    println!("{},{},{},{}", x, y, abc, c);

    // unbinding
    let (a,b) = (10,"Something");
    println!("{},{}", a, b);

    // output using index
    println!("{0},{1},{0}", a, b);

    // a constant is always immutable
    const CON:f32 = 123.; // a constant, while declaration needs typed efined along with it
    println!("{0}",CON );

    // arrays
    let arr:[i32;5] = [1,3,6,8,2]; // it has fixed size

    // tuple can have different data type
    let _s = String::new();
    let data = "Something else";
    let s = data.to_string();
    println!("{}", s);

    // for loop
    for i in arr.iter(){
        println!("{}",i);
    }

    // ownership
    let x = 10; // x is the owner of 10, in this scope and is destroyed afterwards
    println!("x is the owner of {}", x);
    let y = x; // now y is the owner of 10 
    println!("y is the owner of {}", y);
    println!("{}",x);

    //structures
    let rectangle = Shape{no_of_edges :4, no_of_sides:4};
    println!("The sum of sides and edges is {}",sum(&rectangle));

    // modules
    /*
    Mod defines a private module 
    pub defines a public one
    use is import
    */

    // vectors (list in python)
    let _v:Vec<f64> = Vec::new(); // to create a vector
    let v = vec![1.,2.,3.,8.,2.]; // vector macro to create a vector
    println!("{}",v[3]);

}

fn sum(rect : &Shape) -> i32
{
    return rect.no_of_edges as i32 + rect.no_of_sides as i32; // casting
}
struct Shape
{
    no_of_sides:i8,
    no_of_edges:i64
}
