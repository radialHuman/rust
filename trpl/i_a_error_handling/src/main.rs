use std::fs;
use std::fs::File; // file reading
use std::io;
use std::io::ErrorKind; // different errors
use std::io::Read;

fn main() {
    println!("\n**** ERROR HANDLING ****");
    // recoverable error is like missing file
    // unrecoverable is like access of address beyond array length
    // other languages treat them as one by using exception
    // Rust has
    // 1. type Result<T,E> for recoverable
    // 2. panic! for stopping if unrecoverable

    println!("\n**** PANIC! ****");
    // this prints failure message, unwinds, cleans the stack and quits
    // this is a lot of work, an alternate is ABORT, which will just quit handing over the cleaning task to OS
    // this would make binary small and can be done by adding panic = 'abort' in cargo.toml

    // panic!("Array out of bound");

    let v = vec![1, 4, 6, 3];
    // println!("{}", v[5]); // buffer overread, leades to secturity issues as this shouldn give the information in the address

    println!("\n**** RESULT<T,E>! ****");
    // reading a ghost file
    let f = File::open("new.txt");

    let f = match f {
        Ok(file) => file,
        Err(error) => panic!("An error has occured while reading the file: {}", error),
    };

    println!("\n**** DIFFERENT ERRORS ****");
    // if the error is not only missing file but permission or sometoher reason
    // matches can be sued but unwrap_or_else is a closure that cna be used too to make it more concise
    let f = File::open("new.txt");

    let f = match f {
        Ok(file) => file,
        Err(error) => match error.kind() {
            // checking for the kind of error
            ErrorKind::NotFound => match File::create("new.txt") {
                // creating a file if doesn exist
                Ok(f2) => f2,
                Err(error2) => panic!("Problem in creating the file : {:?}", error2),
            },
            other_error => panic!("Something apart from usual has happened: {:?}", other_error),
        },
    };

    println!("\n**** UNWRAP ****");
    // match statement is verbose
    // this is a wrapper over it to active the same conditions in Result enum, but it doesn show any particular output or message
    // except can be used to make it look better
    // let f = File::open("new1.txt").unwrap();
    let f = File::open("new.txt").expect("Failed to open the file");

    println!("\n**** PROPAGATING ERROR (FUNCTION RETURNING ERRORS) ****");
    // its better to return a error from a function if failed
    // ? can be implemented only with functions that return
    fn_reading_file();
    fn_reading_file_better();
    fn_reading_file_even_better();
    fn_reading_file_best();

    // if a fucntion does not return Result<T, E>
    // 1. either convert it
    // 2. or match to handle

    println!("\n**** BOX<dyn Error> ??? ****");
    // used as return type for main only
    // it is a trait object
    // handles any kind of errors

    // {
    //     fn main() -> Result<(), Box<dyn Error>> {
    //         let f = File::open(new1.txt)?;
    //         Ok(())
    //     }
    // }

    println!("\n**** TO PANIC! OR NOT TO PANIC! ??? ****");
    //
}

fn fn_reading_file_best() -> Result<String, io::Error> {
    fs::read_to_string("new1.txt")
}

fn fn_reading_file_even_better() -> Result<String, io::Error> {
    let mut s = String::new();
    File::open("new1.txt")?.read_to_string(&mut s)?;
    Ok(s)
}

fn fn_reading_file_better() -> Result<String, io::Error> {
    let mut f = File::open("new1.txt")?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s)
}

fn fn_reading_file() -> Result<String, io::Error> {
    let f = File::open("new1.txt");

    // returning the reror
    let mut f = match f {
        Ok(file) => file,
        Err(error) => return Err(error),
    };

    // returning the string hence no ; in the end
    let mut s = String::new();
    match f.read_to_string(&mut s) {
        Ok(_) => Ok(s),
        Err(e) => Err(e),
    }
}

// Result type is a enum having two variants
// enum Result<T, E> {
//     Ok(T),  // type of the value to be returned in case of success
//     Err(E), // type of the error to be returned in case of failure
// }

/*
OUTPUT
**** ERROR HANDLING ****

**** PANIC! ****

**** RESULT<T,E>! ****

**** DIFFERENT ERRORS ****

**** UNWRAP ****

**** PROPAGATING ERROR (FUNCTION RETURNING ERRORS) ****

**** BOX<dyn Error> ??? ****

**** TO PANIC! OR NOT TO PANIC! ??? ****
*/
