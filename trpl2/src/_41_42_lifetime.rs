//

/*
*/

pub fn function() {
    // LIFETIMES
    /*
    > It is generally implied {in scope existence} like types but sometimes mentioned particularlly using generic notations
    > A borrow checker is used to check the references
    */
    let r;
    {
        let s = 12;
        // r = &s;// CORRECTED TO
        r = s;
    } // beyond this point, s does not exist sp r cant refere to nothing (dangling)
    println!("{}", r);

    /*
    > in this case, life time of r is from 12 to end of program
    > While that of s is till line 16. Since it ends before r's ending it cant be used
    */
    // In case of functions
    let a = "Some";
    let b = "Something";
    // if all the paramters in and out of the function are & then annotation is important
    // println!(
    //     "{} is the longest of {} and {}",
    //     longest_string_fails(a, b),
    //     a,
    //     b
    // );
    println!(
        "{} is the longest of {} and {}",
        longest_string_with_lifetime_annotation(a, b),
        a,
        b
    );
    /*
    > 'static life time is for &str and is like global
    */
}

// fn longest_string_fails(string1: &str, string2: &str) -> &str {
//     if string1 > string2 {
//         string1
//     } else {
//         string2
//     }
// }

// the annotation gives the same annotation, that means they all live and die together
// if a function has &self, then the output & will have the same lifetime
fn longest_string_with_lifetime_annotation<'a>(string1: &'a str, string2: &'a str) -> &'a str {
    if string1 > string2 {
        string1
    } else {
        string2
    }
}

// struct can also have lifetime
struct SomeStruct<'a> {
    s: &'a str,
}

/*
OUTPUT
12
Something is the longest of Some and Something
*/
