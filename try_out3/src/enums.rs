#[allow(dead_code)]
enum Pay {
    Cash,
    Card,
    // added later
    Bitcoin,
    // redundant addition
    Newspaper,
    Soap,
    Acid,
}

// enums can also have associated paramteres, modifying the previous one
#[allow(dead_code)]
enum PayAmt {
    Cash(f32),
    Card(String, f32), // these types can be complex as structs too
    Crypto { crypto_type: String, amt: f32 }, // this also is valid
}
pub fn code() {
    println!("**** ENUMS ****",);
    let collection = Pay::Cash;
    match collection {
        Pay::Cash => println!("Payed using Cash",),
        Pay::Card => println!("Payed using Card",),
        // at this point if a new element is added to enum pay, and is not covered in this, it will throw an error
        // this is useful for exhaustive check
        Pay::Bitcoin => println!("Payed using Bitcoin",),
        // for not including something or other elements
        _ => {}
    }

    // amount in cash
    let collection = PayAmt::Crypto {
        crypto_type: "BitCoin".to_string(),
        amt: 1000.,
    };
    match collection {
        PayAmt::Cash(x) => println!("{} Payed using Cash", x),
        // variable can also be left unused by adding _ instead of the variable name
        PayAmt::Card(_, x) => println!("{} Payed using Card", x),
        PayAmt::Crypto { crypto_type, amt } => println!("{} Payed using {}", amt, crypto_type),
    }
}

/*
OUTPUT

*/
