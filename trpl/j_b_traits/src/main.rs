fn main() {
    // traits: used to share and assign functions to types/structs
    let someone = Person {
        name: "Some one".to_string(),
        weight: 30.,
        height: 5.,
    };
    println!("{}'s BMI is {}", someone.name, someone.calculate_bmi(1.)); // variable calls functions
    println!("Weight reset to 10 is {}", Person::reset_weight(10.)); // :: because no &self in paramter of the function, Struct calls function
}
pub struct Person {
    pub name: String,
    pub weight: f64,
    pub height: f64,
}
impl Person {
    fn calculate_bmi(&self, offset: f64) -> f64 {
        self.weight / self.height + offset
    }
    fn reset_weight(offset: f64) -> f64 {
        offset
    }
}
