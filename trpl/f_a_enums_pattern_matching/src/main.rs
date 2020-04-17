// at times enums are better than structs
// if a variable can take only one state at a time, ex ip address

enum IPkind {
    V4,
    V6,
}

#[derive(Debug)]
enum IPkindType {
    V4(String),
    V6(String),
}

struct StructType {
    kind: IPkind,
    address: String,
}

fn main() {
    println!("**** ENUMS ****");
    // both are of same data type and can be used as parameter type in fucntion
    let four = IPkind::V4;
    let six = IPkind::V6;

    ipfinder(four);
    ipfinder(six);

    // enum can be used to reduce the types in structs by adding type after it in () ex : IPkindType, StructType
    // with struct
    let _struct_example = StructType {
        kind: IPkind::V4,
        address: "127.0.0.1".to_string(),
    };

    // with enum
    let enum_example = IPkindType::V4(String::from("127.0.0.1"));
    println!("{:?}", enum_example);

    /* advantages
    Any kind and amount of data can be stored
    Can do what in a single enum instead of multiple structs, makes it easier to use it in functions
    */

    println!("**** OPTIONS ****");
    /* option enums (instead of null)
    1. used to check for bugs, since RUST does not have null, this enum helps checking
    2. some and none can be called with out option::_
    */
    let some_number = Some(5);
    let some_string = Some("Something");

    let nothing: Option<i32> = Option::None; // None has to be type specific
}

// enum
enum Option<T> {
    // variants:
    Some(T),
    None,
}

enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}

fn ipfinder(_ip: IPkind) {}

/*
OUTPUT
*/
