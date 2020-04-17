// these or any avoiding of warnings are not to be used in production
#[allow(unused_variables)]
// #[allow(unused_assignments)]
#[allow(non_snake_case)]

// public function to be accessed via main
pub fn code() {
    println!("**** BOOLEANS ****");
    let boolean_1 = true; // auto function like in cpp is enabled and type specificaiton is not required
    let Boolean_2: bool = false; // not a snake case, cariables should be in snake case
    println!("{} is {}", boolean_1, Boolean_2);

    // variables, by default, are immutable
    // boolean_1 = false; // this will produce error
    println!("\n**** MUTABLILITY ****");
    let mut a = false;
    println!("Mutable a was {}", a);
    a = true;
    println!("a is now {}", a);

    // various integer types based on size of the variable
    println!("\n**** INTEGERS ****");
    let x: i8 = 127; // 2^8 with range from -ve to +ve ie. -128 to 127
                     // i32 is the default
                     // u is unsigned (only positive) use case : RBG colors
    println!(
        "Max of u8 is {} and the min is {}",
        std::u8::MAX,
        std::u8::MIN
    );
    println!(
        "Max of u16 is {} and the min is {}",
        std::u16::MAX,
        std::u16::MIN
    );
    println!(
        "Max of u32 is {} and the min is {}",
        std::u32::MAX,
        std::u32::MIN
    );
    println!(
        "Max of u64 is {} and the min is {}",
        std::u64::MAX,
        std::u64::MIN
    );
    println!(
        "Max of u128 is {} and the min is {}",
        std::u128::MAX,
        std::u128::MIN
    );

    println!(
        "Max of i8 is {} and the min is {}",
        std::i8::MAX,
        std::i8::MIN
    );
    println!(
        "Max of i16 is {} and the min is {}",
        std::i16::MAX,
        std::i16::MIN
    );
    println!(
        "Max of i32 is {} and the min is {}",
        std::i32::MAX,
        std::i32::MIN
    );
    println!(
        "Max of i64 is {} and the min is {}",
        std::i64::MAX,
        std::i64::MIN
    );
    println!(
        "Max of i128 is {} and the min is {}",
        std::i128::MAX,
        std::i128::MIN
    );

    // the problem araises when this is not caught during typing but panics when in debug mode
    // let mut y = 2;
    // y = x + y;
    // println!("{} is the new value of y", y); // this will lead to an error : panicked at 'attempt to add with overflow'

    // to avoid this and see what the value which the variable takes, run cargo run --release

    // decimal values
    println!("\n**** FLOATS ****");
    // f64 is default
    let x: f32 = 127.;
    println!(
        "Max of f32 is {} and the min is {}",
        std::f32::MAX,
        std::f32::MIN
    );

    // chars, 4 bytes as it is more than just ascii
    println!("\n**** Character ****");
    let c = 'C';
    println!("{} is a character", c);

    // decimals are different than floats, they will be covered with strings and datetime later as they are not primitives and can be used using crates
}

/*
OUTPUT
**** BOOLEANS ****
true is false

**** MUTABLILITY ****
Mutable a was false
a is now true

**** INTEGERS ****
Max of u8 is 255 and the min is 0
Max of u16 is 65535 and the min is 0
Max of u32 is 4294967295 and the min is 0
Max of u64 is 18446744073709551615 and the min is 0
Max of u128 is 340282366920938463463374607431768211455 and the min is 0
Max of i8 is 127 and the min is -128
Max of i16 is 32767 and the min is -32768
Max of i32 is 2147483647 and the min is -2147483648
Max of i64 is 9223372036854775807 and the min is -9223372036854775808
Max of i128 is 170141183460469231731687303715884105727 and the min is -170141183460469231731687303715884105728

**** FLOATS ****
Max of f32 is 340282350000000000000000000000000000000 and the min is -340282350000000000000000000000000000000

**** Character ****
C is a character
*/
