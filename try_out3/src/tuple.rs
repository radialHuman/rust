pub fn code() {
    println!("**** TUPLE ****");
    let tup = (1, 2, 3., 4, 5, 'R', "Something".to_string(), (1.5, 365)); // heterogenous combination
    println!("{} is in tuple {:?}", tup.0, tup);
    // to access tuple within tuple
    println!("{} is in a tuple within the tuple {:?}", tup.7 .1, tup); // the space is required, elese there will be a complier error ???
    println!(
        "{} is also in a tuple within the tuple {:?}",
        (tup.7).0,
        tup
    );

    // from rbg function
    println!("The color combinaiton returned is {:?}", get_rbg());
    println!("R:{} B:{} G:{}", get_rbg().0, get_rbg().1, get_rbg().2);

    // empty tuple/ unit tuple
    let mt_tupl = (); // can be used in the end of a match where the branch need not do anything
    let num = 10;
    match num {
        0..=9 => println!("single digit"),
        _ => mt_tupl,
    }
    // unit tuple is also the return value of a procedure, often found in the erro message
}

// function to get RBG value
fn get_rbg() -> (u8, u8, u8) {
    // since rbg values are always positive or 0
    (240, 123, 56)
}

/*
OUTPUT
**** TUPLE ****
1 is in tuple (1, 2, 3.0, 4, 5, 'R', "Something", (1.5, 365))
365 is in a tuple within the tuple (1, 2, 3.0, 4, 5, 'R', "Something", (1.5, 365))
1.5 is also in a tuple within the tuple (1, 2, 3.0, 4, 5, 'R', "Something", (1.5, 365))
The color combinaiton returned is (240, 123, 56)
R:240 B:123 G:56
*/
