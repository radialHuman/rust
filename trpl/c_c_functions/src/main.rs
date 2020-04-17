fn main() {
    println!("\n**** FUNCTIONS ****",);
    // main is the entry point to the program
    // fn is used to declare functions, using snake case i.e. fn some_function()
    // the location of function body in {} doesn mater as long as it exists
    //
    some_function(42, "string".to_string());

    println!("\n**** STATEMENTS vs EXPRESSIONS ****",);
    // RUST is a expression-based language
    // unlike other languages difference is important
    // Statements are instructions that perform some action and do not return a value
    // Expressions evaluate to a resulting value, and do not end with ;
    // x= y =6; is invalid in RUST
    // assignments like y = 6  does not return anything hence it is a statement
    let y = {
        let x = 4; // accessible only in this block and not in main, gets cleaned
        x * 2 // this is the value of the block and gets assigned to y
    };
    println!("Value of y is {}", y);

    println!("The square of 5 is {}", function_that_returns(5));
}
fn some_function(i: i32, s: String) {
    println!("{} and {} are the function parameter/aruguments", i, s);
}

fn function_that_returns(i: i32) -> i32 {
    // return type declared
    i * i // no ; since it is just an expression, if ; is placed it will assume the default return () is to be returned which will raise a mismatch error
}

/*
OUTPUT
**** FUNCTIONS ****
42 and string are the function parameter/aruguments

**** STATEMENTS vs EXPRESSIONS ****
Value of y is 8
The square fo 5 is 25
*/
