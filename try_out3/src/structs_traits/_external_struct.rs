pub struct External {
    // this has to be upper camel case
    pub int: i32,
    pub string: String,
}
// the functions are outside the struct and can be implemented anywhere even in the main

// methods can be a part of structs too, these are associated functions
impl External {
    pub fn external_functions1(a: i32, b: &str) -> Self {
        // capital Self is a type, while self is a data
        // this means the output is supposed to be the struct , the function is associated to
        println!("{} and {} are from the external associated function", a, b);
        Self {
            int: a,
            string: b.to_string(),
        }
    }
    pub fn external_functions2(&self, c: i32) -> bool {
        self.int > c
    }
}

// traits is the solution to polymorphism in Rust
pub trait NewTrait {
    // Pascal casing
    fn is_valid(&self) -> i32;
    // fn get_the+better1(&self, other_number : &Self) -> Self;
}

// function using the trait
impl NewTrait for External {
    // traits are implied to be public
    fn is_valid(&self) -> i32 {
        self.int
    }
}

// a unistruct is used to group functionalities together, it has no fields
// generic structs, like templates in cpp
// struct tuple
