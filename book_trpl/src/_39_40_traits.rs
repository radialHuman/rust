//

/*
*/

pub struct Bike {
    cc: f64,
    gears: u8,
    wheel_size: f64,
    tank: u8,
}

impl CarsNBikes for Bike {
    fn customized_vehicle_ratio(&self) -> f64 {
        ((self.cc + self.tank as f64) / self.gears as f64) + self.wheel_size
    }
}

pub struct Car {
    cc: f64,
    gears: u8,
    wheel_size: f64,
    tank: u8,
    windows: u8,
}

impl CarsNBikes for Car {
    fn customized_vehicle_ratio(&self) -> f64 {
        ((self.cc + self.tank as f64) / self.gears as f64) - self.windows as f64 + self.wheel_size
    }
}

// to calculate cc to gear ratio is a function shared between the above two structs
pub trait CarsNBikes {
    fn customized_vehicle_ratio(&self) -> f64;
    // this will contain only the function introduction not the definition
    // definition will be customized as per stucts/enum
    // the body might vary but the return type, paramters remain the same

    // if the functionality is the same then it can be implemented here like:
    fn common_vehicle_ratio(&self) {
        println!("{}", self.customized_vehicle_ratio());
    }
    // trait that can be used by other functions
    fn print_something(&self) {
        println!("This is from the trait passed on as a parameter to this function",);
    }
}

// // passing trait as parameter to a function
// pub fn function_with_trait(item: impl CarsNBikes) {
//     item.print_something();
// }
// // another way of writting it TRAIT BOUND
// pub fn function_with_trait<T: CarsNBikes + std::fmt::Display>(item: T) {
//     item.print_something();
// }
// // another way is using where
// pub fn function_with_trait<T, U>(item1: T, item2: U)
// where
//     T: CarsNBikes + Display,
//     U: CarsNBikes + Debug,
// {
// }

pub fn function() {
    // TRAITS
    /*
    > Shared behaviour to a type
    > trait bounds can be used to define a generic with the behaviour
    > Example is two structs with different features, but need a common behaviour
    > This can be done using trait
    OPHAN RULE/ COHERENCE
    > If a trait has to be implemented, either the trait or the type has to be local to the crate
    > Else rust will get confused

    > Same implementation of a trait can be done with different traits passed into it and will be caleed only when the traits wihtin as satisfied
    */
    // creating a bike
    let rancher = Bike {
        cc: 250.,
        gears: 5,
        wheel_size: 15.5,
        tank: 5,
    };
    // creating a car
    let newton = Car {
        cc: 3000.,
        gears: 56,
        wheel_size: 10.5,
        tank: 15,
        windows: 4,
    };

    // customized trait
    println!(
        "The vehicle ratio of Rancher is : {}",
        rancher.customized_vehicle_ratio()
    );
    println!(
        "The vehicle ratio of Newton is : {}",
        newton.customized_vehicle_ratio()
    );

    // common function
    rancher.common_vehicle_ratio();
    newton.common_vehicle_ratio();

    // function with trait as parameter
    // function_with_trait(rancher);
    function1(&vec![64, 43, 68, 90, 12, 55, 80]);
    function1(&vec![1., 5.2, 3.6, 55.2, 0.36]);
    function1(&vec!['r', '3', 's', 'a', '!', '?']);
    function1(&vec!["w3e", ",", "sadsfds", "a2q", "34"]);
}

fn function1<T>(arr: &Vec<T>)
where
    T: PartialOrd + std::fmt::Debug + std::fmt::Display, // solving < issue
{
    let mut largest = &arr[0];
    for i in arr.iter() {
        // the < in next line will not work until a trait is mentioned to know how it would repond in case of types that dont have clear understanding of <
        if largest < i {
            largest = i;
        }
    }
    // similar to <, displaying of output will not be understood untill a trait is mentioned on how to act
    println!("{} is the largest in {:?}", largest, arr);
}

/*
*/
