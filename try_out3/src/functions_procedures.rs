// these or any avoiding of warnings are not to be used in production
// #[allow(unused_variables)]
// #[allow(unused_assignments)]
#[allow(non_snake_case)]

// public function to be accessed via main
pub fn code() {
    // functions return a value and procedures dont
    // technically the main functions is actually a procedure
    println!("\n**** FUNCTIONS ****");
    println!(
        "{} is the sum from function",
        function1(23., 128, String::from("something"))
    );
    procedure1(23., 128);

    string_procedure("A har"); // literal
    let slice = &"A hardcoded literal";
    string_procedure(slice); // slice
    let string = "A hardcoded literal";
    string_procedure(&string); // String coerced to be slice

    string_procedure2(string.to_string()); // unlike other languages, from now on the string variable is no longer usable in this function as it has been passed on
                                           // this will not be a problem for slices
                                           // This is due to OWNERSHIP AND BORROWING
}

fn function1(a: f32, b: i128, _c: String) -> f32 {
    // _ before the parameter means ignore for time being
    a + b as f32
}

fn procedure1(a: f32, b: i128) {
    // no arrow as nothing is expected
    println!("{} is the sum from procedure", a + b as f32);
}

fn string_procedure(a: &str) {
    if a.len() > 5 {
        println!("only {} was received from {}", &a[..5], a);
    } else {
        println!("only {} was received from {}", &a[..2], a);
    }
}

fn string_procedure2(a: String) {
    println!("only {} was received from {}", &a[..10], a);
}

/*
OUTPUT
**** FUNCTIONS ****
151 is the sum from function
151 is the sum from procedure
only A  was received from A har
only A har was received from A hardcoded literal
only A har was received from A hardcoded literal
only A hardcode was received from A hardcoded literal
*/
