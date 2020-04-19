// https://app.pluralsight.com/course-player?clipId=afec5751-67bd-4328-bc9f-21e2712e7519

use std::any::type_name;

fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

pub fn function() {
    // SLICES
    let arr = [1; 10];
    let slice = &arr[2..5];
    println!(
        "The slice of {:?} is {:?} with type {}",
        arr,
        slice,
        type_of(slice)
    );
    // STRING
    // &str i s like a string view with utf-8 characters
    // indexing cant be done
    let s = "Something";
    for i in s.chars() {
        print!("{},", i);
    }
    println!("",);
    for i in s.chars().rev() {
        print!("{},", i);
    }
    println!("",);
    // to really do indexing
    if let Some(index) = s.chars().nth(2) {
        println!("The 2nd letter is {}", index);
    }

    // STRING is a heap datatype
    let mut letter = String::new();
    let mut a = 'a' as u8;
    while a <= ('z' as u8) {
        letter.push(a as char);
        letter.push(',');
        a += 1;
    }
    println!("The all alphabet string is {}", letter);

    let concat1 = letter.clone() + "1234567890";
    println!("Concatenation of String and &str {}", concat1); // String and &str concat
    let concat2 = letter.clone() + &letter;
    println!("Concatenation of String and String {}", concat2); // String and String concat

    // manipulating &str
    let mut x = "Some thing".to_string(); // or String::from("Some thing");
    x.push_str("!!!");
    println!("{}", x);

    // TUPLE
    // destructuring
    let (a, b) = tuple_maker(12, 17);
    println!(
        "The values from function is {} (sum) and {} (is 12>17)",
        a, b
    );
    // PATTERN MATCHING
    // GENERICS
}

fn tuple_maker(a: i32, b: i32) -> (i32, bool) {
    (a + b, a > b)
}

/*
OUTPUT
*/
