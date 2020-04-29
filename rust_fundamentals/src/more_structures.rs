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
    let (a, b) = tuple_maker(100, 16);
    println!(
        "The values from function is {} (sum) and {} (is 12>17)",
        a, b
    );
    // PATTERN MATCHING
    let a_string = match a {
        0 => "no",
        1 | 2 => "one or two",
        _ if (a > 12) & (a % 2 == 0) => "Even number",
        3..=11 => "a few",
        12 => "a dozen",
        range @ 100..=10000 => "More than enough", // naming the range
        _ => "a lot",
    };
    println!("Match of {} is {}", a, a_string);

    // GENERICS
}

fn tuple_maker(a: i32, b: i32) -> (i32, bool) {
    (a + b, a > b)
}

/*
OUTPUT
The slice of [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] is [1, 1, 1] with type &[i32]
S,o,m,e,t,h,i,n,g,
g,n,i,h,t,e,m,o,S,
The 2nd letter is m
The all alphabet string is a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,
Concatenation of String and &str a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,1234567890
Concatenation of String and String a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,
Some thing!!!
The values from function is 28 (sum) and false (is 12>17)
Match of 28 is Even number
*/
