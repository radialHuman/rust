// useful functions collections

// to find the type of any variable
use std::any::type_name;

fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}
//===============================================================================================================================================

// a neuron without activation fucntion
#[derive(Debug)]
struct NodeInput {
    input: Vec<f64>,
    weights: Vec<f64>,
    bias: f64,
}

impl NodeInput {
    pub fn dot_product(&self) -> f64 {
        let output: f64 = self
            .input
            .iter()
            .zip(self.weights.iter())
            .map(|(x, y)| x * y)
            .sum();
        output + self.bias
    }
}
//===============================================================================================================================================
extern crate map_vec; // for having set

//===============================================================================================================================================
i.unwrap().to_string().parse().unwrap(); // from usize to str to string to int

//===============================================================================================================================================
use rand::Rng;
rand::thread_rng().gen_range(1, 101); // geenrating random number

//===============================================================================================================================================
// reading the input
io::stdin()
        .read_line(&mut input) // storing it in input and avoiding copying by using reference
        .expect("Failed to read input"); // in case the operation fails, The output type of previous line, Result, is handelled with this

//===============================================================================================================================================
// converting string to int
&input.trim().parse::<f64>();

//===============================================================================================================================================
fn generic_fucntion_for_largest_in_string_or_vector<T: PartialOrd + Copy>(l: &[T]) -> T {
    // adding in the traits to T
}

//===============================================================================================================================================
let f = File::open("some_file_that_doesnt_exist.txt");
// to open a file

//===============================================================================================================================================

