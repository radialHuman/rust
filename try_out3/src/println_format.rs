pub fn code() {
    println!("**** PRINTLN ****",);
    // println! is marco for line by line
    // print! is macro for without \n in the end
    println!("Some",);
    println!("Thing",);

    print!("Some",);
    print!("Thing\n",);

    // {} are for inserting tvalues
    let (a, b, c) = (1, 2, 3);
    println!("{} is {} plus {}", c, b, a); // as per the placement of variables
    println!("{2} is {1} plus {0}", a, b, c); // hardcoded order
    println!(
        "{total} is {add1} plus {add2}",
        total = c,
        add1 = a,
        add2 = b
    ); // named variables

    // complicated variables
    let data = NewStruct {
        int: 32,
        float: 12.5,
    };
    println!("{:?} is data", data);
    println!("{:#?} is data", data); // pretty print

    //format
    println!("**** FORMAT ****",);
    let printed = format!("Complex data structure is {0:#?}", data); // makes this a String
    println!("{} is a variable", printed);

    //marco
    println!("**** MACRO ****",);
    // there are various macros like ! (declarative) and the one above the struct below # (Procedural)
    // these generate codes when summoned
}

#[derive(Debug)] // this allows println to print complex datatypes
struct NewStruct {
    int: i32,
    float: f32,
}

/*
OUTPUT
**** PRINTLN ****
Some
Thing
SomeThing
3 is 2 plus 1
3 is 2 plus 1
3 is 1 plus 2
NewStruct { int: 32, float: 12.5 } is data
NewStruct {
    int: 32,
    float: 12.5,
} is data
**** FORMAT ****
COmplex data structure is NewStruct {
    int: 32,
    float: 12.5,
} is a variable
*/
