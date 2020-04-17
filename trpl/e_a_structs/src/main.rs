#[derive(Debug)]
struct Person {
    // upper camel case
    name: String, // String is prefered over &str
    age: u32,
    height: f32,
    alive: bool,
}
// METHODS
// used in context with struct or enum
// just like function but returns a value
// and has self as parameter
// simpler syntax than function, no repeatition
// nice way to organize structs and theier implementation
impl Person {
    fn height_into_age(&self) -> f32 {
        // no ownership taken, if required &mut can be used
        self.height * self.age as f32
    }
}

fn main() {
    println!("\n**** STRUCTS ****");
    // structs or structures are custome datatypes made out of builtin types
    // same as tuples but each element is named for clarity, hence more flexible
    // defined like normal types
    let mut someone = Person {
        // the whole variable is mutable / immutable, cant be semi
        name: "Some One".to_string(),
        age: 32,
        height: 6.4,
        alive: false,
    };
    // to update a value
    someone.name = "Some 1".to_string(); // possible if its mutable

    // calling a function returning struct
    let s2 = make_person(String::from("some 2"), 21);
    println!(
        "{0}({3}) is {1} years old and {2} feet high ",
        s2.name, s2.age, s2.height, s2.alive
    );

    // using one struct to create another struct
    let _some2 = Person {
        // the whole variable is mutable / immutable, cant be semi
        name: "Some Two".to_string(),
        age: 42,
        height: someone.height,
        alive: someone.alive,
    };
    // OR
    let some3 = Person {
        // the whole variable is mutable / immutable, cant be semi
        name: "Some Three".to_string(),
        age: 32,
        ..someone
    };

    // tuple struts are structs without names
    // unit structs dont have any fields, and are usually used to make traits

    // printing struct object can be done by #[derive(Debug)] over struct
    println!("{:#?}", s2);

    // METHODS
    println!("\n**** METHODS ****");
    println!("{} is the height into age result", some3.height_into_age());

    // Associated fucntions are with stucts but dont use self and starts with impl
    // Methods can have same name with different parameter
}

fn make_person(n: String, a: u32) -> Person {
    Person {
        name: n,
        age: a,
        height: 0.0,
        alive: false,
    }
}

/*
OUTPUT
**** STRUCTS ****
some 2(false) is 21 years old and 0 feet high
Person {
    name: "some 2",
    age: 21,
    height: 0.0,
    alive: false,
}

**** METHODS ****
204.8 is the height into age result
*/
