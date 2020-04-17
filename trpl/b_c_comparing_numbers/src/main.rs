// comparing the number entered vs number generated
use rand::Rng;
use std::cmp::Ordering;
use std::io;

fn main() {
    // from the previous main file
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

    // comparing the input with secret number, if input is small it selects Ordering::Less
    match input.cmp(&number) {
        Ordering::Equal => println!("The numbers match!",),
        Ordering::Less => println!("The secret number is bigger!",),
        Ordering::Greater => println!("The secret number is smaller!",),
    }
}

/*
OUTPUT
Game to see if you can guess a random number between 1-100!
45
Missed, it was 52! try again!
The secret number is bigger!
*/
