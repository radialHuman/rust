fn main() {
    println!("\n**** DATA TYPES ****",);
    // every variable has a datatype which helps rust deal with it in particular way
    // there are scalar and compound types

    // SCALAR
    // int has signed and unsigned of 8,16,32(default),64,128 and architecure based
    // int also has decimal, hex, octal, binay and u8
    // floats have 32 and 64(default) as per IEEE-754
    // boolean is bool: true and flase
    // char type has ascii and more like smileys etc

    // COMPOUND
    // tuple is heterogeneous data type collection with fixed length
    let tup: (i32, f64, u8) = (500, 6.4, 1);
    println!("TUPLE {:?} has {}, {}, {}", tup, tup.0, tup.1, tup.2);
    // array is heterogeneous data type collection with fixed length
    let a = [1, 2, 3, 4, 5]; // or
    let a: [i32; 5] = [1, 2, 3, 4, 5];
    println!("ARRAY {:#?}", a); // this a pretty print
    let b = [3; 5]; // repeating 3 5 times
    println!("ARRAY {:?}", b);
    println!("{:?} has {}", a, a[0]);
    // a[10] // which does not exist will raise runtime error, which is unique to RUST as others allow access to memory in that location
    // vectors are the dynamic counterpart of arrays

    // since RUST is a statically typed language, types of all variable must be known at complie time
    // complier can also infer the type without specifying like auto in cpp
    // if something can hvae multiple types then parse is used
    let x: u32 = "42".parse().expect("Not a number"); // must have type mentioned
    println!("{} is from a string 42", x);
}

/*
OUTPUT
**** DATA TYPES ****
TUPLE (500, 6.4, 1) has 500, 6.4, 1
ARRAY [
    1,
    2,
    3,
    4,
    5,
]
ARRAY [3, 3, 3, 3, 3]
[1, 2, 3, 4, 5] has 1
42 is from a string 42
*/
