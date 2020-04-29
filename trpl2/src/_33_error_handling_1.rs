// 9.2
//

use std::fs::File; // to read a local file
use std::io::ErrorKind;

pub fn function() {
    // a program does not have to be stopped if some error occurs
    // such cases can be handelled by Result type which is a tyep of Enum
    // which returns T if okay else e an error, these are generic type
    let f = File::open("some_file_that_doesnt_exist.txt");
    // since this file does not exist this might lead to error but can be dealt in a better way using result
    let f = match f {
        Ok(t) => println!("The file was read successfully {:?}", t),
        Err(e) => panic!("There was an error : {}", e),
    };

    // in case of different kind of error, like lack of permission or file doesn exist etc the respetive error has to be displayed
    let f = File::open("some_file_that_doesnt_exist.txt");
    let f = match f {
        Ok(t) => t,
        Err(e) => match e.kind() {
            ErrorKind::NotFound => match File::create("some_file_that_doesnt_exist.txt") {
                Ok(file_created) => file_created,
                Err(e) => panic!("Created but had error : {:?}", e),
            },
            other_error => panic!("Its a very problematic file:{:?}", other_error),
        },
    };

    // another easier, experienced way to do the same is using unwrap_or_else with clousers
    // matches are powerful but primitive
    let f = File::open("some_file_doest_exist.txt").unwrap_or_else(|error| {
        if error.kind() == ErrorKind::NotFound {
            File::create("some_file_doesnt_exist.txt").unwrap_or_else(|error| {
                panic!("Creating failed {:?}", error);
            })
        } else {
            panic!("Problem opening file {:?}", error);
        }
    });
}
