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
// To transpose a matrix 
pub fn transpose<T>(matrix: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let mut output = vec![];
    for j in 0..matrix[0].len() {
        for i in 0..matrix.len() {
            output.push(matrix[i][j]);
        }
    }
    let x = matrix[0].len();
    shape_changer(&output, matrix.len(), x)
}
//===============================================================================================================================================
// use a list to make a matrix
pub fn shape_changer<T>(list: &Vec<T>, columns: usize, rows: usize) -> Vec<Vec<f64>> {
    /*Changes a list to desired shape matrix*/
    // println!("{},{}", &columns, &rows);
    let mut l = list.clone();
    let mut output = vec![vec![]; rows];
    for i in 0..rows {
        output[i] = l[..columns].iter().cloned().collect();
        // remove the ones pushed to putput
        l = l[columns..].iter().cloned().collect();
    }
    output
}

//===============================================================================================================================================
// Matrix multiplication

pub fn matrix_product(input: &Vec<Vec<f64>>, weights: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    if input.len() == weights.len() {
        // println!(
    //     "Multiplication of {}x{} and {}x{}",
    //     input.len(),
    //     input[0].len(),
    //     weights.len(),
    //     weights[0].len()
    // );
    // println!("Weights transposed to",);
    let weights_t = transpose(&weights);
    // print_a_matrix(&weights_t);
    let mut output: Vec<f64> = vec![];
    for i in input.iter() {
        for j in weights_t.iter() {
            // println!("{:?}x{:?},", i, j);
            output.push(dot_product(&i, &j));
        }
    }
    // println!("{:?}", output);
    shape_changer(&output, input.len(), weights_t.len())
    } else {
        println!("The matrix is invalid");
        vec![]
    }
}

//===============================================================================================================================================
// Dot product
pub fn dot_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
}

//===============================================================================================================================================
// Vector addition

pub fn vector_addition(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    if a.len() == b.len() {
        let mut output = vec![];
        for i in 0..a.len() {
            output.push(a[i] + b[i]);
        }
        output
    } else {
        println!("The vector is invalid",);
        vec![]
    }
}

//===============================================================================================================================================
// To sleep

use std::thread;
use std::time::Duration;
thread::sleep(Duration::from_secs(2));
//===============================================================================================================================================
