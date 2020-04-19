// https://app.pluralsight.com/courses/a7c1879f-d826-499a-9207-63854cf5e932/table-of-contents

use std::mem;

pub fn function() {
    // STRUCTS
    // ENUMS
    // OPTION & if let

    // option gives one of two outputs either some value or none
    let x = 10.;
    let y = 0.1; // change to 0. if want to flip the outcome
    let result: Option<f64> = if y != 0. { Some(x / y) } else { None };
    // to get the value if of an option
    match result {
        Some(x) => println!("The output is {}", x),
        None => println!("Denominator was 0"),
    };

    // variable initialization inside a control structure like if and while
    // instead of if result == Some(x)  {...} else {}
    if let Some(x) = result {
        println!("If let says {}", x);
    }
    // there is no else statement, ergo if denominator is 0, this statement will not be valid

    // ARRAYS
    let mut a: [i32; 5]; // declaration
    a = [1, 5, 2, 8, 0];
    // size is known in advance and is fixed and cant be increased
    // a[5] = 10;
    println!("Length of array is {} starting with {}", a.len(), a[0]);
    a[0] = 10;
    println!("Now the first value is {}", a[0]);
    // to show the array in one go without loop
    println!("The array is {:?}", a);
    // repeat
    let b = [1; 10];
    println!("The repeatition is {:?}", b);
    // range in for loop to change values
    let mut c = [1; 10];
    for i in 0..10 {
        c[i] = i as i32 + 1;
    }
    println!("The previous array is now {:?}", c);
    // memory used by array
    println!(
        "The space occupied by the previous array is {} bytes",
        mem::size_of_val(&c)
    );
    // reducing size of array
    let mut c = [1u16; 10];
    for i in 0..10 {
        c[i] = i as u16 + 1;
    }
    println!(
        "The space occupied by the previous array changes to {} bytes",
        mem::size_of_val(&c)
    );
    // multidimensional array
    let matrix = [[1., 2., 5.], [0.4, 0.6, 3.]];
    println!("The matrix is {:?}", matrix);

    // VECTORS
    // better than array as it is variable in size
    let mut v1 = vec![1, 4, 2, 6, 7];
    println!("Vector {:?}", v1);
    v1.push(10);
    println!("Modified vector {:?}", v1);
    v1.pop(); // also gives option, incase there is nothing in the vector
    println!(
        "Back to original vector {:?} with first number {}",
        v1,
        v1[0] // 0 here becomes usize and not int
    );
    // if length of a vector is unknown, use get for an option
    match v1.get(6) {
        Some(x) => println!("6th element is {}", x),
        None => println!("Try something smaller",),
    }
    // pop in reverse order and show
    while let Some(x) = v1.pop() {
        println!("{}", x);
    }
}

/*
OUTPUT
The output is 100
If let says 100
Length of array is 5 starting with 1
Now the first value is 10
The array is [10, 5, 2, 8, 0]
The repeatition is [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
The previous array is now [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
The space occupied by the previous array is 40 bytes
The space occupied by the previous array changes to 20 bytes
The matrix is [[1.0, 2.0, 5.0], [0.4, 0.6, 3.0]]
Vector [1, 4, 2, 6, 7]
Modified vector [1, 4, 2, 6, 7, 10]
Back to original vector [1, 4, 2, 6, 7] with first number 1
Try something smaller
7
6
2
4
1
*/
