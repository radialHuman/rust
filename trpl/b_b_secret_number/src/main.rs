// a random number is being generator and the user can guess
// there is no inbuilt function to generate but one can be found in https://crates.io/ and has to be added in toml's dependencies
// after updating the toml file, cargo build has to be run, this downloads and updates registry
// also the cargo.lock file gets updated automatically
// all these crates can be updated using cargo update

use rand::Rng;
use std::io; // for user input reading, like in b_a // from the random generator crate

fn main() {
    println!("Game to see if you can guess a random number between 1-100!",);
    let secret_number = rand::thread_rng().gen_range(1, 101); // geenrating random number

    // asking user to enter
    // creating a mutable variable to store user input as a string
    let mut input = String::new();

    // reading the input
    io::stdin()
        .read_line(&mut input) // storing it in input and avoiding copying by using reference
        .expect("Failed to read input"); // in case the operation fails, The output type of previous line, Result, is handelled with this

    // converting input to number
    let number = secret_number.to_string();

    // checking if the user has guessed it right
    if input == number {
        println!("You have guess it right!",);
    } else {
        println!("Missed, it was {}! try again!", secret_number);
    }

    // printing random numbers
    for _ in 1..100 {
        print!("{},", rand::thread_rng().gen_range(1, 101));
    }
}

/*
OUTPUT
Game to see if you can guess a random number between 1-100!
47
Missed, it was 22! try again!
67,21,3,50,100,28,93,87,61,34,15,51,82,68,24,77,76,47,5,92,53,20,78,80,12,41,84,5,84,22,78,84,57,25,86,82,44,64,83,64,73,66,71,13,31,45,32,5,77,54,33,78,88,85,77,2,99,96,43,34,11,21,94,29,11,18,57,52,57,37,22,52,21,45,54,41,76,73,29,92,3,57,26,85,49,35,71,20,75,56,56,15,20,72,1,46,67,17,19,
*/
