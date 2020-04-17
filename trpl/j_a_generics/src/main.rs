fn main() {
    // generics do not cost during runtime due to monomorphism
    println!("\n**** FUCNTIONS ****");

    let v1 = vec![1, 5, 3, 7, 2, 9, 3, 6, 4, 5, 4, 3, 4];
    println!(
        "{} is the largest",
        generic_fucntion_for_largest_in_string_or_vector(&v1)
    );

    let v1 = ["som", "ewr", "dm", "lh", "xz", "xx", "!!"];
    println!(
        "{} is the largest",
        generic_fucntion_for_largest_in_string_or_vector(&v1)
    );

    println!("\n**** STRUCTS ****");

    let s1 = GenericStruct {
        x: 1,
        y: "Something",
    };
    println!("{:?}", s1);

    let s2 = GenericStruct {
        x: String::from("Some"),
        y: "thing",
    };
    println!("{:?}", s2);

    println!("\n**** TRAITS ****");
    // define what a type can do

    // **** Traits and lifetime incomplete ****
}

#[derive(Debug)]
struct GenericStruct<T, U> {
    x: T,
    y: U,
}

fn generic_fucntion_for_largest_in_string_or_vector<T: PartialOrd + Copy>(l: &[T]) -> T {
    // adding in the traits to T
    let mut largest = l[0];
    for &i in l.iter() {
        if i > largest {
            largest = i;
        }
    }
    largest
}

/*
OUTPUT
**** FUCNTIONS ****
9 is the largest
xz is the largest

**** STRUCTS ****
GenericStruct { x: 1, y: "Something" }
GenericStruct { x: "Some", y: "thing" }
*/
