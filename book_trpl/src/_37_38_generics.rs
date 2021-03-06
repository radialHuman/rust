//

/*
Generics for functions enums and structs
*/

// struct can also use generic types
pub struct NameAge<T, U> {
    name: T,
    age: U,
}
// this method is for any type
impl<T: std::fmt::Debug, U: std::fmt::Debug> NameAge<T, U> {
    pub fn returncombo(&self) -> String {
        format!("{:?}-{:?}", self.name, self.age)
    }
    pub fn adding_new_type_inside_the_function<V>(&self, random_number: V) -> V {
        random_number
    }
}
// excusive methods also can be made
impl NameAge<String, String> {
    pub fn return_combo(&self) -> String {
        format!("{:?}_{:?}", self.name, self.age)
    }
}

// same for enums

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
    let someone = NameAge {
        name: String::from("Some"),
        age: 1,
    };
    println!("The values from structs: {},{}", someone.name, someone.age);
    println!("The someone's id is {}", someone.returncombo());

    // exculsive
    let someone_else = NameAge {
        name: String::from("Some"),
        age: "32".to_string(),
    };
    println!(
        "The someoneElse's exclusive id is {}",
        someone_else.return_combo()
    );
    println!(
        "New generic type inside implementation {}",
        someone_else.adding_new_type_inside_the_function(2.3)
    )
}

// // Left incomplete in this part due to trait implementation
// fn function1<T, U>(arr: &Vec<T>) -> U {
//     let mut largest = arr[0];
//     for i in arr.iter() {
//         // the < in next line will not work until a trait is mentioned to know how it would repond in case of types that dont have clear understanding of <
//         if largest < *i {
//             largest = *i;
//         }
//     }
//     // similar to <, displaying of output will not be understood untill a trait is mentioned on how to act
//     format!("{} is the largest", largest)
// }

/*
99 is the largest
The values from structs: Some,1
The someone's id is "Some"-1
The someoneElse's exclusive id is "Some"_"32"
New generic type inside implementation 2.3
*/
