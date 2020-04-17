use std::io; // for user input, an io lib

fn main() {
    println!("Input a number");

    // creating a mutable variable to store user input
    let mut input = String::new();

    // reading the input
    io::stdin()
        .read_line(&mut input) // storing it in input and avoiding copying by using reference
        .expect("Failed to read input"); // in case the operation fails, The output type of previous line, Result, is handelled with this

    // checking if the entered value is numerical
    let test = &input.trim().parse::<f64>();
    match test {
        Ok(_ok) => println!("The input was {}", input),
        Err(_e) => println!("Enter numbers only, try again!"),
    }
}
