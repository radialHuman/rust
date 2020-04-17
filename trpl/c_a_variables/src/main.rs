fn main() {
    // variables are immutable by default, remain almost always in the scope
    println!("\n**** IMMUTABILITY ****",);
    let x = 5; // can be changed down the line
    println!("Value of immutable x is  {}", x);

    let mut y = 6;
    println!("Value of mutable y was {}", y);
    y = 7;
    println!("Value of mutable y is {}", y);

    // Constants are always immutable, can be declared and used in any part of the code
    // A constant can be set to a constant expression and not to result of a function call or values computed at runtime only
    // Const are useful to have hardcoded values as it makes it easy to change values in future for coders
    println!("\n**** CONSTANTS ****",);
    const PI: f32 = 3.14; // there is no const mut PI, must have data type, capitals
    println!("Value of the constant pi is and will always be {}", PI);

    // Shadowing, declaring the variable again using the keyword let
    // a bit different than mut as it can have transformation involved and in the end the new value of the variable is immutable unline mut
    // also the type of variable can be chaged using let which will raise compiler error in mut
    println!("\n**** LET vs MUT ****",);
    let z = 10;
    println!("Value of z was {}", z);
    let z = "ten";
    println!("Value of z is {}", z);
    // z = "one"; // this will raise error if immutable variable is in use

    let mut w = 10;
    println!("Value of w was {}", w);
    // w = "ten"; // this will raise error as the type is being changed
    w = 11;
    println!("Value of w is {}", w);
}

/*
OUTPUT
**** IMMUTABILITY ****
Value of immutable x is  5
Value of mutable y was 6
Value of mutable y is 7

**** CONSTANTS ****
Value of the constant pi is and will always be 3.14

**** LET vs MUT ****
Value of z was 10
Value of z is ten
Value of w was 10
Value of w is 11
*/
