pub fn code() {
    println!("**** CASTING ****",);
    // there is no implicit casting only explicit in RUST
    println!("{} + {} = {}", 12, 13.5, 12 + 13. as i32);
    println!("{} + {} = {}", 12, '#', 12 + '#' as i32);
    println!("{} + {} = {}", 12, true, 12 + true as i32);
    // this can cause data loss and is not encouraged by RUST
    println!("\n**** SHADOWING ****",);
    let variable = 10;
    println!("{} is the value in main", variable);
    {
        // this cope can be anything liek a conditional or a loop or match etc
        let variable = "Something";
        println!("'{}' is the value in scope 1", variable);
        {
            let variable = 4.2;
            println!("{} is the value in scope 1.1", variable);
        }
    }
    println!("{} is still the value in main", variable);

    println!("\n**** CONSTANTS ****",);
    const PI: f64 = 3.14; // const needs type, and they dont occupy moemory, instead it replaces variable with value in code and complies, ergo increasing runtime speed
    println!(" Inbuilt pi value is {}", std::f32::consts::PI);
    println!(" Constant pi value is {}", PI);
    // Mutable global variable
    println!("\n**** STATIC ****",);
    // this is a dangerous one so rust forces it be used this way :
    unsafe {
        STATIC_GLOBAL_VARIABLE = 123;
        println!(
            "Dangerous shared mutable memory can cause havoc and must eb in the unsafe scope {}",
            STATIC_GLOBAL_VARIABLE
        );
    }
}

static mut STATIC_GLOBAL_VARIABLE: i32 = 234;

/*
OUTPUT
**** CASTING ****
12 + 13.5 = 25
12 + # = 47
12 + true = 13

**** SHADOWING ****
10 is the value in main
'Something' is the value in scope 1
4.2 is the value in scope 1.1
10 is still the value in main

**** CONSTANTS ****
 Inbuilt pi value is 3.1415927
 Constant pi value is 3.14

**** STATIC ****
Dangerous shared mutable memory can cause havoc and must eb in the unsafe scope 123
*/
