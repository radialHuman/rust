//
use std::fs::File;
use std::io;
use std::io::Read;

/* */

pub fn function() {
    // instead of having the error printed out, it can be passed on as a function returing argument
    // this way anyone can use the fucntion and handle the error they way they want to
    // the return is an enum result so either success or error is printed
    println!("{:?}", propagate_error());
    // this lets error messgae printed out without showing error if the code had stopped
}
pub fn propagate_error() -> Result<String, io::Error> {
    let f = File::open("Some file.txt");
    let mut f = match f {
        Ok(s) => s,
        Err(t) => return Err(t),
    };
    let mut s = String::new();
    match f.read_to_string(&mut s) {
        Ok(_) => Ok(s),
        Err(t) => Err(t),
    }
    // the same thing can be replicated by using ?
    // replacing match with ? in the end does the same thing as long as function has returning error
    // let f = File::open("Some file.txt")?; // if problematic it raises the error
    // f.read_to_string(&mut s)?; // else if problematic it raises the error
    // Ok(s) // else sends string

    // better way cna be suing chaining action
    // let f = File::open("Some file.txt")?.f.read_to_string(&mut s)?;

    // the best way to write this after function call is
    // fs::read_to_string("some file.txt");
}

// main can have Result as outptu apart from ()
use std::error::Error;
fn main_proxy() -> Result<(), Box<dyn Error>> {
    let f = File::open("Some file.txt")?;
    Ok(())
}
