#[derive(Debug)] // for printing enums
#[allow(dead_code)] // for not using all the elemnets of the enum
enum Coins {
    Penny,
    Nickel,
    Dime,
    Quarter,
}

#[derive(Debug)]
#[allow(dead_code)]
enum StateCoins {
    Penny,
    Nickel,
    Dime,
    Quarter(State),
}

#[derive(Debug)]
#[allow(dead_code)]
enum State {
    Solid,
    Liquid,
    Gas,
}

fn main() {
    println!("\n**** MATCH ****");
    /*
    1. Powerful control flow statement
    2. compares value against patterns and executes corresponding codes
    3. If returns a bool vs match can be modified to return anything
    4. Matches are exhaustive, if a struct or enum is being dealt with, it should take care of all the cases
    */

    let c = Coins::Dime;
    match c {
        Coins::Penny => println!("{:?} was selected", c),
        Coins::Nickel => println!("{:?} was selected", c),
        Coins::Dime => println!("{:?} was selected", c),
        Coins::Quarter => println!("{:?} was selected", c),
    }

    println!("\n**** MATCH WITH BINDED VALUE ****");
    let c = StateCoins::Quarter(State::Liquid);
    match c {
        StateCoins::Penny => println!("{:?} was selected", c),
        StateCoins::Nickel => println!("{:?} was selected", c),
        StateCoins::Dime => println!("{:?} was selected", c),
        StateCoins::Quarter(State) => println!("Quarter was selected from the {:?} state", State),
    }

    println!("\n**** MATCH WITH OPTIONS ****");
    let six = plus_one(Option::Some(5));
    let na = plus_one(Option::None);
    println!("{:?} and {:?} are from Options", six, na);

    println!("\n**** MATCH WITH PLACEHOLDERS ****");
    // for anythign thats out of the scope of pattern matching
    let x = 100;
    match x {
        1..=10 => println!("{} is less than 11", x),
        1..=20 => println!("{} is less than 21", x),
        1..=30 => println!("{} is less than 31", x),
        1..=40 => println!("{} is less than 41", x),
        1..=50 => println!("{} is less than 51", x),
        _ => println!("{} is more than 51", x),
        // _ => (); // to ignore it and return empty tuple, defualt output of functions
    }

    println!("\n**** IF LET ****");
    // If match is only for one pattern, it can be wordy to use match syntax
    // can also be combined with else
}

#[derive(Debug)]
enum Option<T> {
    None,
    Some(T),
}

fn plus_one(x: Option<i32>) -> Option<i32> {
    match x {
        Option::None => Option::None,
        Option::Some(i) => Option::Some(i + 1),
    }
}

/*
OUTPUT
**** MATCH ****
Dime was selected

**** MATCH WITH BINDED VALUE ****
Quarter was selected from the Liquid state

**** MATCH WITH OPTIONS ****
Some(6) and None are from Options

**** MATCH WITH PLACEHOLDERS ****
100 is more than 51
*/
