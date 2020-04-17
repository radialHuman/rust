// these or any avoiding of warnings are not to be used in production
#[allow(unused_variables)]
// #[allow(unused_assignments)]
#[allow(non_snake_case)]

// public function to be accessed via main
pub fn code() {
    println!("\n**** STRINGS ****");

    // strings are difficult in all the languages but easy to handle
    // but in rust due to the decision on memory and trade-off its difficult to learn
    // complexity is not hidden in RUST when it comes to string
    // benefit of this is runtime speed and concurrency etc.
    // two types of strings
    println!("\n**** String slice ****");
    let string_slice: &str = "Some string slice"; // grouped u8s character which are mostly immutable
    let string: String = String::from("Some string"); // like the strings in other languages, from is one of the ways to create strings

    // the way they are stored and handeled are different even though they store the same thing
    // Strings area on heap hence mutable, holding data for long time
    // &str slice is immutable and can be both on heap or stack or in the code, for faster runtime
    // casting can be done easily
    let string_from_slice: String = string_slice.to_string();
    // making a string literals (has properties of slices) to String
    let hardcoded = "Something hardcoded".to_string(); // definig the type is not required as its auto
                                                       // making a String to slice
    let slice_from_string = &hardcoded; // definig the type is not required as its auto, this is not a copy, but a pointer to the value as it is referenced
                                        // refernced or in RUST borrowing is very efficient for runtime

    // concatenation of string literals, slices and strings
    println!("\n**** String concatenation ****");
    let concatenated_string1 = ["String1", "String2"].concat(); // this returns a String type
    println!("{} is the new string", concatenated_string1);
    // method 2
    let concatenated_string2 = format!("{} {}", "String1", "String2");
    println!("{} is the new string", concatenated_string2);

    // to maka a slice by concatenation
    let slice1 = string_from_slice + string_slice; // the order has to be String + &str else it will lead to error

    // mutable string with New
    let mut new_string = String::new();
    new_string.push_str(slice_from_string);
    new_string.push_str("|Some other string literal");
    new_string.push_str(&"|string: &str");
    new_string.push('!'); // to push chars
    println!("{} is the new string pushed with slices", new_string);

    // adding two strings is done efficiently by taking the first string and the reference of others
    let string1 = String::from("Some");
    let string2 = String::from(" stuff");
    let string3 = String::from(" exists");
    let new_string2 = string1 + &string2 + &string3;
    println!(
        "{} <- is the new string added with slices of others",
        new_string2
    );

    // substring
    println!("\n**** Sub-String ****");
    let sub_string1 = &new_string2[4..15].to_string(); // .. is upto but not including
    let sub_string2 = &new_string2[4..=15].to_string(); // .. is upto
    println!("{} is a part of {}", sub_string1, new_string2);

    // if the index is overflown, there wont be an error but it will crash

    // to get a char out of str
    let char1 = &new_string2.chars().nth(1);
    println!("{:?} is in {}", char1, new_string2);
}

/*
OUTPUT
**** STRINGS ****

**** String slice ****

**** String concatenation ****
String1String2 is the new string
String1 String2 is the new string
Something hardcoded|Some other string literal|string: &str! is the new string pushed with slices
Some stuff exists <- is the new string added with slices of others

**** Sub-String ****
 stuff exis is a part of Some stuff exists
Some('o') is in Some stuff exists
*/
