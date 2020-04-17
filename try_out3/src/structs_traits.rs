mod _external_struct; // calling the external file, this has to be in a seperate folder as its not being called to main
use _external_struct::*; // using the code in it

#[derive(Debug, Clone)] // ???

struct NewStruct {
    // struct name in pascal case, and elements in snake case
    s_name: String,
    s_age: i32,
    s_human: bool,
}

pub fn code() {
    println!("**** STRUCTS ****",);
    // similar to class and objects but a bit different
    // no inheritence, but can have methods
    // traits can help with polymorphism
    // Macros
    let mut some1 = NewStruct {
        s_name: "Some One".to_string(),
        s_age: 32,
        s_human: true,
    };
    some1.s_human = false; // cant be updated if it is immutable
    println!("{} is {} years old", some1.s_name, some1.s_age);

    // creating a new variable based on existing struct
    let some2 = NewStruct {
        s_name: "Some Two".to_string(),
        ..some1 // this implies the rest is same as the called in struct
    };
    println!("{} is also {} years old", some2.s_name, some2.s_age);

    // calling external structs
    let anything = External {
        // even though the file has public code, the fields are private, this provides control over visibility
        int: 36,
        string: "Some thing".to_string(),
    };
    println!(
        "{} has {} in it, from the external file",
        anything.string, anything.int
    );

    // Composition over inheritance ???
    External::external_functions1(10, "String"); // :: operator if Self type
                                                 // self using associative function
    let bool_value = anything.external_functions2(10); // dot operator if &self
    println!("{} is from external function comparator", bool_value);
    // trait function
    let new_int = anything.is_valid();
    println!("{} is from external function trait", new_int);

    // macros ???
}

/*
OUTPUT
**** STRUCTS ****
Some One is 32 years old
Some Two is also 32 years old
Some thing has 36 in it, from the external file
10 and String are from the external associated function
true is from external function comparator
36 is from external function trait
*/
