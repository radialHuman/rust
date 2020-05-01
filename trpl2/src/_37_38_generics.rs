//

/*
*/


// struct can also use generic types
pub struct NameAge<T, U> {
    name: T,
    age: U,
}


pub fn function() {
    // GENERICS
    /*
    Functions that accept multiple concrete types are generic functions
    Like Options, Results, Vec, etc. The data type is not known it still works
    If more than 1 function differes only in the input&output types then this can be used to reduce rewritting of code
    This new function, since generalized, will need traits (behaviours) to be added
    */

    // fucntion for finding the largets in vector
    let s = vec![1, 5, 3, 8, 0, 3, 6, 8, 9, 99];
    let mut largest = s[0];
    for i in s.iter() {
        if largest < *i {
            largest = *i;
        }
    }
    println!("{} is the largest", largest);

    // if this has to be a function for f64 or u8 etc then instead of writting different fucntions,
    // generics can be used as in the function1 where T represents any type can be passed and U is any type cna be returned
    // T and U can be the same (and U can then be replaced with T) or different

    // Generics in struct
    let someone = NameAge{
        name: String::from("Some"),
        age: 1,
    }
}
fn function1<T, U>(arr: &Vec<T>) -> U {
    let mut largest = arr[0];
    for i in arr.iter() {
        // the < in next line will not work until a trait is mentioned to know how it would repond in case of types that dont have clear understanding of <
        if largest < *i {
            largest = *i;
        }
    }
    // similar to <, displaying of output will not be understood untill a trait is mentioned on how to act
    format!("{} is the largest", largest)


    
}
