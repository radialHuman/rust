//

/*
Functional style
Closures
*/

use std::thread; // to command a thread
use std::time::Duration; // for waiting

// lazy eval
struct Cacher<T>
where
    T: Fn(u32) -> u32, // Fn, FnMut, FnOnce
{
    calculation: T,
    value: Option<u32>,
}

pub fn function() {
    /*
    > Using functions as values by passing them in arguments
    > Anonymouse functions
    > can caputre variables from scope
    > Used for small functions useful, is more programmer facing then user
    > annotating the types or parameters and return variable might not be required  inline function
    > But the type is assumed to be the one which uses the closure first
    */
    /*
    EXAMPLE
    to print a number after making the user wait
    */
    let input = 10.5;
    let output = generate_output(input);
    println!("{:?}", output);

    // better way using closures
    let output = |num: f64| {
        println!("Please wait while the data is processed ...",);
        thread::sleep(Duration::from_secs(2)); // making them wait for 2 seconds
        num
    };
    println!("{}", output(23.));

    /*
    MEMORIZATION/LAZY EVAL
    > Store the closure in struct and call only when required
    > Cacher ???
    */

    /*
    CAPTURING VARIABLE FROM ENVIRONMENT
    > This has to store the variable somewhere and is an overhead.
    > TO avoid this use functions instead   
    */
    let x = 4;
    let lambda = |z| z == x;
    let y = 4;
    println!("Closure says {}", lambda(y)); // can access x even though x is not an argument
}

pub fn generate_output(i: f64) -> f64 {
    if i < 21.0 {
        println!("Its less than 21",);
        make_them_wait(i)
    } else {
        println!("Its more than 21",);
        make_them_wait(i)
    }
}

pub fn make_them_wait(i: f64) -> f64 {
    println!("Please wait while the data is processed ...",);
    thread::sleep(Duration::from_secs(2)); // making them wait for 2 seconds
    i
}

/*
OUTPUT
Its less than 21
Please wait while the data is processed ...
10.5
Please wait while the data is processed ...
23
*/
