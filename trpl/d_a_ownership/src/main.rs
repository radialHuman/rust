fn main() {
    // a unique way RUST manages memory
    // guarantees safety without garbage collector
    // ownership, borrowing, slices, references
    println!("\n**** OWNERSHIP ****");
    /* its a simple idea with deep impact
    program has to manage memory while it runs
    there are ways other laguages do, like garbage collector, programmer doing it manually
    RUST does it via a set of rules that complier does at compile time
    this does not affect thr runtime
    MEMORY
    stack stores fixed size variables, is fast as pointers can access easily, LIFO, int, char, float, bool etc
    heap is slow, duplicating in it is costly, used when size is uncertain like string, vectors etc
    managing heap data is why ownership exists can help explain why it works the way it does
    RULE 1. Each value has a variable andits the OWNER
    RULE 2. There can only be one owner at a time
    RULE 3. When the owner goes out of scope, the value is dropped
    */
    {
        let s = "Something";
        println!("s has the value of {} in a scope", s);
    }
    println!(
        "Now that the scope is over, s does not exist anymore, and calling it would lead to error",
    );
    // STRING type
    println!("\n**** STRING ****");
    // stored in HEAP and helps understand how RUST deals with such types especially in STD lib
    // string literals are immutable and are stored in stack
    // but cant be known all the time during runtime, so String
    // string literal to string
    let mut s = String::from("Something");
    println!("s was '{}'", s);
    s.push_str(" else");
    println!("s is '{}' after mutation", s);
    // string is mutable as it is in heap
    // everytime its called a request has to be made to the meory to give space, hence the String::from (requesting from heap)
    // since garbage colelctor is not involved in RUST, doing it manually can be problematic
    // RUST does the out of scope function call called DROP which cleans it up, like RAII in cpp

    // COPY in stack
    let x = 10;
    let y = x;
    println!("Value of y on stack is {}", y);
    println!("Value of x on stack is {}", x);
    // usually in other laguages x and y are both 10 as they are pointing to same location
    // but in RUST y gets a copy of x as its easy to do it on a stack
    // the types that have copy trait, they can use older variables
    // copy trait cant be annotated with drop trait

    // MOVE in HEAP
    let w = String::from("xyz");
    let z = w;
    println!("Value of z on heap is '{}'", z);
    // println!("Value of w on heap is '{}'", w); // raises error "value borrowed here after move"

    // Strings are made up of the following on stack while the actual value on heap
    // 1. pointer that holds memory on heap
    // 2. capacity of the variable
    // 3. length of the variable

    // COPY in HEAP
    // can be done using expensive clone method
    let w = String::from("xyz");
    let z = w.clone();
    println!("Value of z on heap is deep copy of '{}'", z);
    // this will create a seperate copy of memory in heap and 3 parts fo string in stack

    // Ownerships relation with functions
    // similar to moving heap variables to another owner and making the previous owner useless
    // once a heap variable is passed to a function, it can be used beyond the function call
    // to avoid this, a function can return the value back to use it further
    // but a stack variable can be used
    // using fucntions like this can be tedious if multiple things are returned, hence references

    // References & (dereferencing * later)
    println!("\n**** REFERENCE AND BORROWING ****");
    // to avoid moving of variables to functions when passed
    let strg = String::from("Somethings");
    let (len, cap) = length_and_capacity(&strg);
    println!("Length is {} and capacity is {} of {}", len, cap, strg);
    println!(
        "strg is still '{}' and is not consumed by the function as it was referenced or just borrowed",
        strg
    );
    // this allows function to borrow and the ownership still remains with the main
    // hence function cant modify it

    // MUTABLE reference
    // only one mutable reference to a particular piece of data in a particular scope ??? this is not the case anymore ???
    let mut strg = String::from("Somethings");
    let _ref1 = &mut strg;
    let _ref2 = &mut strg; // this should raise an error but its not !!!
    let (len, cap) = wrong_length_and_capacity(&mut strg);
    println!("Length is {} and capacity is {} of {}", len, cap, strg);
    println!(
        "strg is now '{}' and is changed even though it was referenced",
        strg
    );

    // this avoids data race problem at complie time when
    /*
    >=2 pointers access the same data at the same time
    At least 1 of the pointers is being used to write to the data
    There is no mechanism being used to synchronize access to the data
    a mutable ref cant exist when there is a immutable one
    */
    // this also avoids dangling pointers, where data is removed while there is a pointer pointing to it (more in lifetime)

    // *** either but not both of the following: one mutable reference or any number of immutable references ***

    // SLICES
    println!("\n**** SLICES ****");
    // like references they dont have ownership
    // reference to a continuous part of a collection only
    // in case of strgin its a part of the string
    // as using index and parsing is difficult
    // it creates a reference with pointer to the starting of the part of string
    println!("'{}' is a part of '{}'", &strg[..6], strg); // doesn work without &
    println!("'{}' is also a part of '{}'", &strg[3..=18], strg); // including 8th element

    println!(
        "'{}' is the first word in '{}'",
        find_first_word(&strg),
        strg
    );

    // string literal are of type &str which are immutable references , also slices
    // &str as parameter will aloow arguments of both &String and &str type

    // slices in arrays
    let x = [1, 2, 3, 4, 5, 6];
    let arr_slice = &x[..=3];
    for i in arr_slice.iter() {
        print!("{}, ", i);
    }
}

fn find_first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i]; // why is there a return when the function is already sending down below ???
        }
    }
    &s[..] // is this the result if if {} doesn work ???
}

fn length_and_capacity(s: &String) -> (usize, usize) {
    // s.push_str(" are not possible"); // this cant happen and will lead to error, this will need mutability
    (s.len(), s.capacity())
}

fn wrong_length_and_capacity(s: &mut String) -> (usize, usize) {
    s.push_str(" are not possible"); // this cant happen and will lead to error, this will need mutability
    (s.len(), s.capacity())
}
/*
OUTPUT
**** OWNERSHIP ****
s has the value of Something in a scope
Now that the scope is over, s does not exist anymore, and calling it would lead to error

**** STRING ****
s was 'Something'
s is 'Something else' after mutation
Value of y on stack is 10
Value of x on stack is 10
Value of z on heap is 'xyz'
Value of z on heap is deep copy of 'xyz'

**** REFERENCE AND BORROWING ****
Length is 10 and capacity is 10 of Somethings
strg is still 'Somethings' and is not consumed by the function as it was referenced or just borrowed
Length is 27 and capacity is 27 of Somethings are not possible
strg is now 'Somethings are not possible' and is changed even though it was referenced

**** SLICES ****
'Someth' is a part of 'Somethings are not possible'
'ethings are not ' is also a part of 'Somethings are not possible'
'Somethings' is the first word in 'Somethings are not possible'
1, 2, 3, 4,
*/
