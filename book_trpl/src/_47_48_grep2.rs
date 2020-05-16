//

/*
*/
use std::env; // for user input
use std::fmt::Debug; // trait for printing
use std::fs; // to read a file

#[derive(Debug)]
pub struct Input {
    query: String,
    filename: String,
}

impl Input {
    pub fn new_panic(i: &[String]) -> Input {
        // checking the number of parameters
        if i.len() != 3 {
            panic!("\n\n!!! RETRY! Please enter a search word followed by the filepath with extension !!! \n\n");
        }
        let query = i[1].clone();
        let filename = i[2].clone();
        Input { query, filename }
    }
    pub fn new(i: &[String]) -> Result<Input, &'static str> {
        // lifetime for error till the end of the program
        // checking the number of parameters
        if i.len() != 3 {
            return Err("\n\n!!! RETRY! Please enter a search word followed by the filepath with extension !!! \n\n");
        }
        let query = i[1].clone();
        let filename = i[2].clone();
        Ok(Input { query, filename })
    }
}

pub fn function() {
    /*
    > Adding new to the struct, makes it idiomatic and easy to initialize any variable like String::new
    > Adding better error message for (panics are good for programmer not for user, use result instead)
        > Count of input
    */
    let user_entry: Vec<String> = env::args().collect();
    let useful_input = Input::new(&user_entry);
    let content = fs::read_to_string(useful_input.filename).expect("ERROR!!! File not not read");
    println!("{:?} has to be found in {:?}", useful_input.query, content);
}

/*
OUTPUT

*/
