pub fn code() {
    println!("**** IF ****");
    // all the statements are like in other languages, but a small additions in rust
    let bol = false;
    if bol == true {
        // can also be if bol {...} since it is a bool and true if not
        // if !bol {...}
        // the braces should be in the same line
        println!("It is {}", bol);
    } else {
        // else should be this way and not in the next line after the ending if braces
        println!("It is {}", bol);
    }

    // complex conditions
    // the first branch that hits the condition is considered and the rest is discarded
    // if none is hit then it ignores all
    let var = 10;
    if !bol && var < 10 {
        println!("{} is {}", bol, var);
    } else if bol && var > 10 {
        println!("{} is {}", bol, var);
    } else if bol && var <= 10 {
        println!("{} is {}", bol, var);
    } else if !bol && var <= 10 {
        println!("Condition met");
    }

    // inline if
    let inline_var = if bol { 'T' } else { 'F' }; // these act as mini functions that return a value ergo no ; required
    println!("Inline variable is {}", inline_var);

    // match
    match bol {
        true => {
            println!("Inline variable depends on bol, which is {}", bol);
        }
        false => {
            println!("Inline variable depends on bol, which is {}", bol);
        }
    }

    let num = 14;
    match num {
        -100..=-1 | 0 => println!("The number is {}", num), // here or is just a single pipe
        1..=10 => println!("The number is between 1 and 11"),
        11..=100 => println!("The number is between 10 and 101"),
        _ => println!("The number is between beyond 101 or less than 0"),
    }
}

/*
OUTPUT
**** IF ****
It is false
Condition met
Inline variable is F
Inline variable depends on bol, which is false
The number is between 10 and 101
*/
