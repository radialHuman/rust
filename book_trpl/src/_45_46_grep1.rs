//
/*
To create a cmd like environment where a file can be read and printout the sentence where a word exist
*/
use std::env; // to get user input
use std::fs; // to read a file

pub fn function() {
    println!("Enter a file name and a word to be searched in it",);
    // let args: Vec<String> = env::args().collect(); // collect needs type specified
    // println!("Searching for `{}` \nIn {}...", args[1], args[2]);

    // let filename = &args[2];
    // let word = &args[1];

    // let file_content = fs::read_to_string(filename).expect("ERROR!!! File not not read");
    // println!("The content in the file was: \n {}", file_content);

    // Structuring/Organizing the code
    /* RIGHT NOW main does few task, but generally its better to have one idea per fucntion
    for ease of tresting and maintaining
    Its better to have all the variable sin one scope to avoid confusion
    better error to provide info like if the file was missing or its not able to read etc not a simple error like above
    */
    /*
    > Main should call the logic in lib.rs
    > If main grows bigger, trasnfer some to lib
    > Main will contain run for lib.rs
    > And to handle if error occurs
    > Main can be tested so all fucntions go to lib.rs
    */
    let args: Vec<String> = env::args().collect();
    let input = parse_config(&args);
    let file_content = fs::read_to_string(input.filename).expect("ERROR!!! File not not read");
    println!("The content in the file was: \n {}", file_content);
}
pub struct input<'a> {
    query: &'a str,
    filename: &'a str,
}

pub fn parse_config(s: &[String]) -> input {
    input {
        query: &s[1],
        filename: &s[2],
    }
}

/*
OUTPUT
Enter a file name and a word to be searched in it
Searching for`something`
In src\main.rs...
The content in the file was:
 // mod _33_error_handling_1;
// mod _34_error_handling_2;
// mod _35_error_handling_panic;
// mod _36_error_handling_when_panic;
// mod _37_38_generics;
// mod _39_40_traits;
// mod _41_42_lifetime;
// mod _42_43_testing;
// mod _44_control_testing;
mod _45_46_grep1;

fn main() {
    // _33_error_handling_1::function();
    // _34_error_handling_2::function();
    // _35_error_handling_panic::function();
    // _36_error_handling_when_panic::function();
    // _37_38_generics::function();
    // _39_40_traits::function();
    // _41_42_lifetime::function();
    // _42_43_testing::function();
    // _44_control_testing::function();
    _45_46_grep1::function();
}
*/
