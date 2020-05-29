use math::round;
use rand::*;

// use crate::lib_matrix::*;

/*
SOURCE
------
Activation from : https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
Neuron : nnfs

DESCRIPTION
-----------------------------------------
STRUCTS
-------
1. LayerDetails : To create a layer of n_neurons and n_inputs
    > 1. create_weights : To randomly generate n_neurons weights between -1 and 1
    > 2. create_bias : A constant 0 (can be modified if required) vector of n_neurons as bias
    > 3. output_of_layer : activation_function((inputs*weights)-bias)

FUNCTIONS
---------
1. activation_leaky_relu :
    > 1. &Vec<T> to be used as input to funtion
    > 2. alpha to control the fucntion's "leaky" nature
    = 1. Modified Vec<T>
2. activation_relu :
    > 1. &Vec<T> to be used as input to funtion
    = 1. Modified Vec<T>
3. activation_sigmoid :
    > 1. &Vec<T> to be used as input to funtion
    = 1. Modified Vec<T>
4. activation_tanh :
    > 1. &Vec<T> to be used as input to funtion
    = 1. Modified Vec<T>
*/
pub struct LayerDetails {
    pub n_inputs: usize,
    pub n_neurons: i32,
}
impl LayerDetails {
    pub fn create_weights(&self) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        let mut weight: Vec<Vec<f64>> = vec![];
        // this gives transposed weights
        for _ in 0..self.n_inputs {
            weight.push(
                (0..self.n_neurons)
                    .map(|_| round::ceil(rng.gen_range(-1., 1.), 3))
                    .collect(),
            );
        }
        weight
    }
    pub fn create_bias(&self) -> Vec<f64> {
        let bias = vec![0.; self.n_neurons as usize];
        bias
    }
    pub fn output_of_layer(
        &self,
        input: &Vec<Vec<f64>>,
        weights: &Vec<Vec<f64>>,
        bias: &mut Vec<f64>,
        f: fn(input: &Vec<f64>) -> Vec<f64>,
    ) -> Vec<Vec<f64>> {
        let mut mat_mul = transpose(&matrix_multiplication(&input, &weights));
        // println!("input * weights = {:?}", mat_mul);
        let mut output: Vec<Vec<f64>> = vec![];
        for i in &mut mat_mul {
            // println!("i*w {:?}, bias {:?}", &i, &bias);
            output.push(vector_addition(i, bias));
        }
        // println!("Before activation it was {:?}", &output[0]);
        // println!("After activation it was {:?}", activation_relu(&output[0]));
        let mut activated_output = vec![];
        for i in output {
            activated_output.push(f(&i));
        }
        // transpose(&activated_output)
        activated_output
    }
}

pub fn activation_relu<T>(input: &Vec<T>) -> Vec<T>
where
    T: Copy + std::cmp::PartialOrd + std::ops::Sub<Output = T> + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    // ReLU for neurons
    let zero = "0".parse::<T>().unwrap();
    input
        .iter()
        .map(|x| if *x > zero { *x } else { *x - *x })
        .collect()
}

pub fn activation_leaky_relu<T>(input: &Vec<T>, alpha: f64) -> Vec<T>
where
    T: Copy + std::cmp::PartialOrd + std::ops::Mul<Output = T> + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    // Leaky ReLU for neurons, where alpha is multiplied with x if x <= 0
    // to avoid making it completely 0 like in ReLU
    let zero = "0".parse::<T>().unwrap();
    let a = format!("{}", alpha).parse::<T>().unwrap();
    input
        .iter()
        .map(|x| if *x > zero { *x } else { a * *x })
        .collect()
}

pub fn activation_sigmoid<T>(input: &Vec<T>) -> Vec<f64>
where
    T: std::str::FromStr + std::fmt::Debug,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    // Sigmoid for neurons
    input
        .iter()
        .map(|x| 1. / (1. + format!("{:?}", x).parse::<f64>().unwrap().exp()))
        .collect()
}

pub fn activation_tanh<T>(input: &Vec<T>) -> Vec<f64>
where
    T: std::str::FromStr + std::fmt::Debug,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    // TanH for neurons
    input
        .iter()
        .map(|x| {
            (format!("{:?}", x).parse::<f64>().unwrap().exp()
                - (format!("{:?}", x).parse::<f64>().unwrap() * (-1.)).exp())
                / (format!("{:?}", x).parse::<f64>().unwrap().exp()
                    + (format!("{:?}", x).parse::<f64>().unwrap() * (-1.)).exp())
        })
        .collect()
}

/*
DESCRIPTION
-----------------------------------------
STRUCTS
-------
FUNCTIONS
---------
1. coefficient : To find slope(b1) and intercept(b0) of a line
> 1. list1 : A &Vec<T>
> 2. list2 : A &Vec<T>
= 1. b0
= 2. b1

2. convert_and_impute : To convert type and replace missing values with a constant input
> 1. list : A &Vec<String> to be converted to a different type
> 2. to : A value which provides the type(U) to be converted to
> 3. impute_with : A value(U) to be swapped with missing elemets of the same type as "to"
= 1. Result with Vec<U> and Error propagated
= 2. A Vec<uszie> to show the list of indexes where values were missing

3. covariance :
> 1. list1 : A &Vec<T>
> 2. list2 : A &Vec<T>
= 1. f64

4. impute_string :
> 1. list : A &mut Vec<String> to be imputed
> 2. impute_with : A value(U) to be swapped with missing elemets of the same type as "to"
= 1. A Vec<&str> with missing values replaced

5. mean :
> 1. list : A &Vec<T>
= 1. f64

6. read_csv :
> 1. path : A String for file path
> 2. columns : number of columns to be converted to
= 1. HashMap<String,Vec<String>) as a table with headers and its values in vector

7. root_mean_square :
> 1. list1 : A &Vec<T>
> 2. list2 : A &Vec<T>
= 1. f64

8. simple_linear_regression_prediction : // https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/
> 1. train : A &Vec<(T,T)>
> 2. test : A &Vec<(T,T)>
    = 1. Vec<T>

9. variance :
    > 1. list : A &Vec<T>
    = 1. f64

10. convert_string_categorical :
    > 1. list : A &Vec<T>
    > 2. extra_class : bool if true more than 10 classes else less
    = Vec<usize>

11. normalize_vector_f : between [0.,1.]
    > 1. list: A &Vec<f64>
    = Vec<f64>

12. logistic_function_f : sigmoid function
    > 1. matrix: A &Vec<Vec<f64>>
    > 2. beta: A &Vec<Vec<f64>>
    = Vec<Vec<f64>>

13. log_gradient_f :  logistic gradient function
    > 1. matrix1: A &Vec<Vec<f64>>
    > 2. beta: A &Vec<Vec<f64>> // same shape as matrix1
    > 3. matrix2: A &Vec<f64> // target
    = Vec<Vec<f64>>

14. cost_function_f :
    > 1. matrix1: A &Vec<Vec<f64>> // input
    > 2. beta: A &Vec<Vec<f64>> // same shape as matrix1
    > 3. matrix2: A &Vec<f64> // target
    = f64

15. gradient_descent :
    > 1. matrix1: &Vec<Vec<f64>>,
    > 2.beta: &Vec<Vec<f64>>,
    > 3.matrix2: &Vec<Vec<f64>>,
    > 4.learning_rate: f64,
    > 5.coverage_rate: f64,
    = Vec<Vec<f64>>
    = i32

16. logistic_predict
    1. > matrix1: &Vec<Vec<f64>>
    2. > beta: &Vec<Vec<f64>>
    = Vec<Vec<f64>>

17. randomize
    1. > rows : &Vec<f64>
    = Vec<f64>

18. train_test_split
    1. > input: &Vec<f64>
    2. > percentage: f64
    = Vec<f64>
    = Vec<f64>

19. binary_logistic_regression
    1. path: String
    2. target_name: String
    3. test_percentage: f64
    4. learning_rate : f64
    5. coverage_rate : f64
    = beta : Vec<Vec<f64>>
    = # of iterations : i32

*/

// use crate::lib_matrix;
// use lib_matrix::*;

pub fn mean<T>(list: &Vec<T>) -> f64
where
    T: std::iter::Sum<T>
        + std::ops::Div<Output = T>
        + Copy
        + std::str::FromStr
        + std::string::ToString
        + std::ops::Add<T, Output = T>
        + std::fmt::Debug
        + std::fmt::Display
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let zero: T = "0".parse().unwrap();
    let len_str = list.len().to_string();
    let length: T = len_str.parse().unwrap();
    (list.iter().fold(zero, |acc, x| acc + *x) / length)
        .to_string()
        .parse()
        .unwrap()
}

pub fn variance<T>(list: &Vec<T>) -> f64
where
    T: std::iter::Sum<T>
        + std::ops::Div<Output = T>
        + std::marker::Copy
        + std::fmt::Display
        + std::ops::Sub<T, Output = T>
        + std::ops::Add<T, Output = T>
        + std::ops::Mul<T, Output = T>
        + std::fmt::Debug
        + std::string::ToString
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let zero: T = "0".parse().unwrap();
    let mu = mean(list);
    let _len_str: T = list.len().to_string().parse().unwrap(); // is division is required
    let output: Vec<_> = list
        .iter()
        .map(|x| (*x - mu.to_string().parse().unwrap()) * (*x - mu.to_string().parse().unwrap()))
        .collect();
    // output
    let variance = output.iter().fold(zero, |a, b| a + *b); // / len_str;
    variance.to_string().parse().unwrap()
}

pub fn covariance<T>(list1: &Vec<T>, list2: &Vec<T>) -> f64
where
    T: std::iter::Sum<T>
        + std::ops::Div<Output = T>
        + std::fmt::Debug
        + std::fmt::Display
        + std::ops::Add
        + std::marker::Copy
        + std::ops::Add<T, Output = T>
        + std::ops::Sub<T, Output = T>
        + std::ops::Mul<T, Output = T>
        + std::string::ToString
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let mu1 = mean(list1);
    let mu2 = mean(list2);
    let zero: T = "0".parse().unwrap();
    let _len_str: T = list1.len().to_string().parse().unwrap(); // is division is required
    let tupled: Vec<_> = list1.iter().zip(list2).collect();
    let output = tupled.iter().fold(zero, |a, b| {
        a + ((*b.0 - mu1.to_string().parse().unwrap()) * (*b.1 - mu2.to_string().parse().unwrap()))
    });
    output.to_string().parse().unwrap() // / len_str
}

pub fn coefficient<T>(list1: &Vec<T>, list2: &Vec<T>) -> (f64, f64)
where
    T: std::iter::Sum<T>
        + std::ops::Div<Output = T>
        + std::fmt::Debug
        + std::fmt::Display
        + std::ops::Add
        + std::marker::Copy
        + std::ops::Add<T, Output = T>
        + std::ops::Sub<T, Output = T>
        + std::ops::Mul<T, Output = T>
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let b1 = covariance(list1, list2) / variance(list1);
    let b0 = mean(list2) - (b1 * mean(list1));
    (b0.to_string().parse().unwrap(), b1)
}

pub fn simple_linear_regression_prediction<T>(train: &Vec<(T, T)>, test: &Vec<(T, T)>) -> Vec<T>
where
    T: std::iter::Sum<T>
        + std::ops::Div<Output = T>
        + std::fmt::Debug
        + std::fmt::Display
        + std::ops::Add
        + std::marker::Copy
        + std::ops::Add<T, Output = T>
        + std::ops::Sub<T, Output = T>
        + std::ops::Mul<T, Output = T>
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let train_features = &train.iter().map(|a| a.0).collect();
    let test_features = &test.iter().map(|a| a.1).collect();
    let (offset, slope) = coefficient(train_features, test_features);
    let b0: T = offset.to_string().parse().unwrap();
    let b1: T = slope.to_string().parse().unwrap();
    let predicted_output = test.iter().map(|a| b0 + b1 * a.0).collect();
    let original_output: Vec<_> = test.iter().map(|a| a.0).collect();
    println!(
        "RMSE: {:?}",
        root_mean_square(&predicted_output, &original_output)
    );
    predicted_output
}

pub fn root_mean_square<T>(list1: &Vec<T>, list2: &Vec<T>) -> f64
where
    T: std::ops::Sub<T, Output = T>
        + Copy
        + std::ops::Mul<T, Output = T>
        + std::ops::Add<T, Output = T>
        + std::ops::Div<Output = T>
        + std::string::ToString
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    println!("========================================================================================================================================================");
    let zero: T = "0".parse().unwrap();
    let tupled: Vec<_> = list1.iter().zip(list2).collect();
    let length: T = list1.len().to_string().parse().unwrap();
    let mean_square_error = tupled
        .iter()
        .fold(zero, |b, a| b + ((*a.1 - *a.0) * (*a.1 - *a.0)))
        / length;
    let mse: f64 = mean_square_error.to_string().parse().unwrap();
    mse.powf(0.5)
}

// reading in files for multi column operations
use std::collections::HashMap;
use std::fs;
pub fn read_csv<'a>(path: String) -> (Vec<String>, Vec<Vec<String>>) {
    println!("========================================================================================================================================================");
    println!("Reading the file ...");
    let file = fs::read_to_string(&path).unwrap();
    let splitted: Vec<&str> = file.split("\r\n").collect();
    let rows: i32 = (splitted.len() - 1) as i32;
    println!("Number of rows = {}", rows - 1);
    let table: Vec<Vec<_>> = splitted.iter().map(|a| a.split(",").collect()).collect();
    let values = table[1..]
        .iter()
        .map(|a| a.iter().map(|b| b.to_string()).collect())
        .collect();
    let columns: Vec<String> = table[0].iter().map(|a| a.to_string()).collect();
    (columns, values)
}

use std::io::Error;
pub fn convert_and_impute<U>(
    list: &Vec<String>,
    to: U,
    impute_with: U,
) -> (Result<Vec<U>, Error>, Vec<usize>)
where
    U: std::cmp::PartialEq + Copy + std::marker::Copy + std::string::ToString + std::str::FromStr,
    <U as std::str::FromStr>::Err: std::fmt::Debug,
{
    println!("========================================================================================================================================================");
    // takes string input and converts it to int or float
    let mut output: Vec<_> = vec![];
    let mut missing = vec![];
    match type_of(to) {
        "f64" => {
            for (n, i) in list.iter().enumerate() {
                if *i != "" {
                    let x = i.parse::<U>().unwrap();
                    output.push(x);
                } else {
                    output.push(impute_with);
                    missing.push(n);
                    println!("Error found in {}th position of the vector", n);
                }
            }
        }
        "i32" => {
            for (n, i) in list.iter().enumerate() {
                if *i != "" {
                    let string_splitted: Vec<_> = i.split(".").collect();
                    let ones_digit = string_splitted[0].parse::<U>().unwrap();
                    output.push(ones_digit);
                } else {
                    output.push(impute_with);
                    missing.push(n);
                    println!("Error found in {}th position of the vector", n);
                }
            }
        }
        _ => println!("This type conversion cant be done, choose either int or float type\n Incase of string conversion, use impute_string"),
    }

    (Ok(output), missing)
}

pub fn impute_string<'a>(list: &'a mut Vec<String>, impute_with: &'a str) -> Vec<&'a str> {
    println!("========================================================================================================================================================");
    list.iter()
        .enumerate()
        .map(|(n, a)| {
            if *a == String::from("") {
                println!("Missing value found in {}th position of the vector", n);
                impute_with
            } else {
                &a[..]
            }
        })
        .collect()
}

// use std::collections::HashMap;
pub fn convert_string_categorical<T>(list: &Vec<T>, extra_class: bool) -> Vec<f64>
where
    T: std::cmp::PartialEq + std::cmp::Eq + std::hash::Hash + Copy,
{
    println!("========================================================================================================================================================");
    let values = unique_values(&list);
    if extra_class == true && values.len() > 10 {
        println!("The number of classes will be more than 10");
    } else {
        ();
    }
    let mut map: HashMap<&T, f64> = HashMap::new();
    for (n, i) in values.iter().enumerate() {
        map.insert(i, n as f64 + 1.);
    }
    list.iter().map(|a| map[a]).collect()
}

pub fn normalize_vector_f(list: &Vec<f64>) -> Vec<f64> {
    // println!("========================================================================================================================================================");
    let (minimum, maximum) = min_max_f(&list);
    let range: f64 = maximum - minimum;
    list.iter().map(|a| 1. - ((maximum - a) / range)).collect()
}

pub fn logistic_function_f(matrix: &Vec<Vec<f64>>, beta: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    println!("========================================================================================================================================================");
    //https://www.geeksforgeeks.org/understanding-logistic-regression/
    println!("logistic function");
    println!(
        "{:?}x{:?}\n{:?}x{:?}",
        matrix.len(),
        matrix[0].len(),
        beta.len(),
        beta[0].len()
    );
    matrix_multiplication(matrix, beta)
        .iter()
        .map(|a| a.iter().map(|b| 1. / (1. + ((b * -1.).exp()))).collect())
        .collect()
}

pub fn log_gradient_f(
    matrix1: &Vec<Vec<f64>>,
    beta: &Vec<Vec<f64>>,
    matrix2: &Vec<f64>,
) -> Vec<Vec<f64>> {
    println!("========================================================================================================================================================");
    //https://www.geeksforgeeks.org/understanding-logistic-regression/
    println!("Log gradient_f");
    // PYTHON : // first_calc = logistic_func(beta, X) - y.reshape(X.shape[0], -1)
    let mut first_calc = vec![];
    for (n, i) in logistic_function_f(matrix1, beta).iter().enumerate() {
        let mut row = vec![];
        for j in i.iter() {
            row.push(j - matrix2[n]);
        }
        first_calc.push(row);
    }

    let first_calc_T = transpose(&first_calc);
    let mut X = vec![];
    for j in 0..matrix1[0].len() {
        let mut row = vec![];
        for i in matrix1.iter() {
            row.push(i[j]);
        }
        X.push(row);
    }

    // PYTHON : // final_calc = np.dot(first_calc.T, X)
    let mut final_calc = vec![];
    for i in first_calc_T.iter() {
        for j in X.iter() {
            final_calc.push(dot_product(&i, &j))
        }
    }

    // println!("{:?}\n{:?}", &first_calc_T, &X);
    // println!("{:?}", &final_calc);
    // println!(
    //     "{:?}",
    //     shape_changer(&final_calc, matrix1[0].len(), matrix1.len())
    // );
    shape_changer(&final_calc, matrix1[0].len(), matrix1.len())
}

pub fn cost_function_f(matrix1: &Vec<Vec<f64>>, beta: &Vec<Vec<f64>>, matrix2: &Vec<f64>) -> f64 {
    println!("========================================================================================================================================================");
    //https://www.geeksforgeeks.org/understanding-logistic-regression/
    // PYTHON: // log_func_v = logistic_func(beta, X)
    // println!(" matrix1 {:?}", matrix1);
    // println!(" beta {:?}", beta);
    // println!(" matrix2 {:?}", matrix2);
    println!(
        "shape\ninput: {:?},{:?}\nbeta: {:?},{:?}\ntarget: {:?}",
        matrix1[0].len(),
        matrix1.len(),
        beta[0].len(),
        beta.len(),
        matrix2.len()
    );
    println!("Calculating cost function ...");
    let logistic_func_v = logistic_function_f(&transpose(&matrix1), &beta);
    let log_logistic: Vec<Vec<f64>> = logistic_func_v
        .iter()
        .map(|a| a.iter().map(|a| a.ln()).collect())
        .collect();
    // println!(" Log logistic {:?}", log_logistic);
    // // PYTHON: // step1 = y * np.log(log_func_v)
    let mut step1 = vec![];
    for i in log_logistic.iter() {
        let mut row = vec![];
        for (n, j) in i.iter().enumerate() {
            for (m, k) in matrix2.iter().enumerate() {
                if n == m {
                    row.push(j * k);
                } else {
                    ()
                }
            }
        }
        step1.push(row);
    }
    let one_minus_matrix2: Vec<f64> = matrix2.iter().map(|b| 1. - b).collect();
    // println!(" 1-y {:?}", one_minus_matrix2);
    let one_minus_log_logistic: Vec<Vec<f64>> = logistic_func_v
        .iter()
        .map(|a| a.iter().map(|b| (1. - b).ln()).collect())
        .collect();
    // println!("one_minus_log_logistic\n{:?}", one_minus_log_logistic);

    let minus_step1: Vec<Vec<f64>> = step1
        .iter()
        .map(|a| a.iter().map(|b| *b * -1.).collect())
        .collect();
    //PYTHON : // step2 = (1 - y) * np.log(1 - log_func_v)
    let mut step2 = vec![];
    for i in one_minus_log_logistic.iter() {
        // println!("{:?}\n{:?}", i, one_minus_matrix2);
        // println!("DONE 2 ISSUE HERE");
        step2.push(element_wise_operation(i, &one_minus_matrix2, "Mul"));
    }

    let minus_step2: Vec<Vec<f64>> = step2
        .iter()
        .map(|a| a.iter().map(|b| *b * -1.).collect())
        .collect();
    // PYTHON : // -step1 -step2
    let mut output = element_wise_matrix_operation(&minus_step1, &step2, "Sub");
    let sum = output
        .iter()
        .fold(0., |a, b| a + b.iter().fold(0., |a, b| a + b));
    sum / (beta.len() * beta.len()) as f64
}

pub fn gradient_descent(
    matrix1: &Vec<Vec<f64>>,
    beta: &mut Vec<Vec<f64>>,
    matrix2: &Vec<f64>,
    learning_rate: f64,
    coverage_rate: f64,
) -> (Vec<Vec<f64>>, i32) {
    let mut cost = cost_function_f(matrix1, beta, matrix2);
    println!("Gradient descent ...");
    let mut iterations = 1;
    let mut change_cost = 1.;
    let mut log_beta: Vec<Vec<f64>> = vec![];
    let mut b: Vec<Vec<f64>> = vec![];
    while change_cost > coverage_rate {
        let old_cost = cost;
        println!("{:?}x{:?}", beta.len(), beta[0].len());
        *beta = element_wise_matrix_operation(
            beta,
            &log_gradient_f(matrix1, beta, matrix2)
                .iter()
                .map(|a| a.iter().map(|b| b * learning_rate).collect())
                .collect(),
            "Sub",
        );
        // println!("=\n{:?}", &beta);
        cost = cost_function_f(matrix1, &beta, matrix2);
        // println!("cost = {:?}", cost);
        change_cost = old_cost - cost;
        // println!("change cost = {:?}", old_cost - cost);
        iterations += 1;
    }
    let output = beta.clone();
    (output, iterations)
}

pub fn logistic_predict(matrix1: &Vec<Vec<f64>>, beta: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    // https://www.geeksforgeeks.org/understanding-logistic-regression/
    let prediction_probability = logistic_function_f(matrix1, beta);
    let output = prediction_probability
        .iter()
        .map(|a| a.iter().map(|b| if *b >= 0.5 { 1. } else { 0. }).collect())
        .collect();
    output
}

pub fn randomize(rows: &Vec<f64>) -> Vec<f64> {
    use rand::seq::SliceRandom;
    use rand::{thread_rng, Rng};
    let mut order: Vec<usize> = (0..rows.len() - 1 as usize).collect();
    let slice: &mut [usize] = &mut order;
    let mut rng = thread_rng();
    slice.shuffle(&mut rng);
    // println!("{:?}", slice);

    let mut output = vec![];
    for i in order.iter() {
        output.push(rows[*i].clone());
    }
    output
}

pub fn train_test_split(input: &Vec<f64>, percentage: f64) -> (Vec<f64>, Vec<f64>) {
    // shuffle
    let data = randomize(input);
    // println!("{:?}", data);
    // split
    let test_count = (data.len() as f64 * percentage) as usize;
    // println!("Test size is {:?}", test_count);

    let test = data[0..test_count].to_vec();
    let train = data[test_count..].to_vec();
    (train, test)
}

pub fn binary_logistic_regression(
    path: String,
    target_name: String,
    test_percentage: f64,
    learning_rate: f64,
    coverage_rate: f64,
) -> (Vec<Vec<f64>>, i32) {
    // use std::collections::HashMap;
    let (columns, values) = read_csv(path);
    // converting input to str and normalizing them
    let mut df: HashMap<String, Vec<f64>> = HashMap::new();
    for (n, i) in columns.iter().enumerate() {
        let mut v = vec![];
        for j in values.iter() {
            for (m, k) in j.iter().enumerate() {
                if n == m {
                    v.push(k.parse().unwrap());
                }
            }
        }
        v = normalize_vector_f(&v);
        df.insert(i.to_string(), v);
    }
    // print!("{:?}", df);
    // test and train split, target and features split
    let mut test_features: HashMap<String, Vec<f64>> = HashMap::new();
    let mut train_features: HashMap<String, Vec<f64>> = HashMap::new();
    let mut test_target: HashMap<String, Vec<f64>> = HashMap::new();
    let mut train_target: HashMap<String, Vec<f64>> = HashMap::new();

    for (k, v) in df.iter() {
        if *k.to_string() != target_name {
            test_features.insert(k.clone(), train_test_split(v, test_percentage).1);
            train_features.insert(k.clone(), train_test_split(v, test_percentage).0);
        // X
        } else {
            test_target.insert(k.clone(), train_test_split(v, test_percentage).1);
            train_target.insert(k.clone(), train_test_split(v, test_percentage).0);
            // y
        }
    }
    let feature_vector: Vec<_> = train_features.values().cloned().collect();
    let target_vector: Vec<_> = train_target.values().cloned().collect();
    let feature_length = feature_vector[0].len();
    // println!("{:?}", target_vector);

    // initiating beta values
    let mut beta_df = HashMap::new();
    for (n, i) in columns.iter().enumerate() {
        let mut v = vec![0.; feature_length];
        beta_df.insert(i.to_string(), v);
    }

    let mut beta = vec![vec![0.; train_features.keys().len()]];
    println!("BETA: {:?}", beta);

    // gradient descent on beta
    let (new_beta, iteration_count) =
        gradient_descent(&feature_vector, &mut beta, &target_vector[0], 0.01, 0.001);
    // println!(
    //     "{:?}\n{:?}\n{:?}\n{:?}\n{:?}",
    //     feature_vector, target_vector, &beta, &new_beta, iteration_count
    // );
    (new_beta, iteration_count)
}

/*
DESCRIPTION
-----------------------------------------
STRUCTS
-------
FUNCTIONS
---------
1. dot_product :
    > 1. A &Vec<T>
    > 2. A &Vec<T>
    = 1. T

2. element_wise_operation : for vector
    > 1. A &mut Vec<T>
    > 2. A &mut Vec<T>
    > 3. operation &str ("Add","Sub","Mul","Div")
    = 1. Vec<T>

3. matrix_multiplication :
    > 1. A &Vec<Vec<T>>
    > 2. A &Vec<Vec<T>>
    = 1. Vec<Vec<T>>

4. pad_with_zero :
    > 1. A &mut Vec<T> to be modified
    > 2. usize of number of 0s to be added
    = 1. Vec<T>

5. print_a_matrix :
    > 1. A &str as parameter to describe the matrix
    > 2. To print &Vec<Vec<T>> line by line for better visual
    = 1. ()

6. shape_changer :
    > 1. A &Vec<T> to be converter into Vec<Vec<T>>
    > 2. number of columns to be converted to
    > 3. number of rows to be converted to
    = 1. Vec<Vec<T>>

7. transpose :
    > 1. A &Vec<Vec<T>> to be transposed
    = 1. Vec<Vec<T>>

8. vector_addition :
    > 1. A &Vec<T>
    > 2. A &Vec<T>
    = 1. Vec<T>

9. make_matrix_float :
    > 1. input: A &Vec<Vec<T>>
    = Vec<Vec<f64>>

10. make_vector_float :
    > 1. input: &Vec<T>
    = Vec<f64>

11. round_off_f :
    > 1. value: f64
    > 2. decimals: i32
    = f64

12. unique_values : of a Vector
    > 1. list : A &Vec<T>
    = 1. Vec<T>

13. value_counts :
    > 1. list : A &Vec<T>
    = HashMap<T, u32>

14. is_numerical :
    > 1. value: T
    = bool

15. min_max_f :
    > 1. list: A &Vec<f64>
    = (f64, f64)

16. type_of : To know the type of a variable
    > 1. _
    = &str

17. element_wise_matrix_operation : for matrices
    > 1. matrix1 : A &Vec<Vec<T>>
    > 2. matrix2 : A &Vec<Vec<T>>
    > 3. fucntion : &str ("Add","Sub","Mul","Div")
    = A Vec<Vec<T>>

18. matrix_vector_product_f
    > 1. matrix: &Vec<Vec<f64>>
    > 2. vector: &Vec<f64>
    = Vec<f64>
*/

pub fn print_a_matrix<T: std::fmt::Debug>(string: &str, matrix: &Vec<Vec<T>>) {
    // To print a matrix in a manner that resembles a matrix
    println!("{}", string);
    for i in matrix.iter() {
        println!("{:?}", i);
    }
    println!("");
    println!("");
}

pub fn shape_changer<T>(list: &Vec<T>, columns: usize, rows: usize) -> Vec<Vec<T>>
where
    T: std::clone::Clone,
{
    /*Changes a list to desired shape matrix*/
    // println!("{},{}", &columns, &rows);
    let mut l = list.clone();
    let mut output = vec![vec![]; rows];
    if columns * rows == list.len() {
        for i in 0..rows {
            output[i] = l[..columns].iter().cloned().collect();
            // remove the ones pushed to output
            l = l[columns..].iter().cloned().collect();
        }
        output
    } else {
        panic!("!!! The shape transformation is not possible, check the values entered !!!");
        // vec![]
    }
}

pub fn transpose<T: std::clone::Clone + Copy>(matrix: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    // to transform a matrix
    let mut output = vec![];
    for j in 0..matrix[0].len() {
        for i in 0..matrix.len() {
            output.push(matrix[i][j]);
        }
    }
    let x = matrix[0].len();
    shape_changer(&output, matrix.len(), x)
}

pub fn vector_addition<T>(a: &mut Vec<T>, b: &mut Vec<T>) -> Vec<T>
where
    T: std::ops::Add<Output = T> + Copy + std::fmt::Debug + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    // index wise vector addition
    let mut output = vec![];
    if a.len() == b.len() {
        for i in 0..a.len() {
            output.push(a[i] + b[i]);
        }
        output
    } else {
        // padding with zeros
        if a.len() < b.len() {
            let new_a = pad_with_zero(a, b.len() - a.len());
            println!("The changed vector is {:?}", new_a);
            for i in 0..a.len() {
                output.push(a[i] + b[i]);
            }
            output
        } else {
            let new_b = pad_with_zero(b, a.len() - b.len());
            println!("The changed vector is {:?}", new_b);
            for i in 0..a.len() {
                output.push(a[i] + b[i]);
            }
            output
        }
    }
}

pub fn matrix_multiplication<T>(input: &Vec<Vec<T>>, weights: &Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Copy + std::iter::Sum + std::ops::Mul<Output = T>,
{
    // Matrix multiplcation
    // println!(
    //     "Multiplication of {}x{} and {}x{}",
    //     input.len(),
    //     input[0].len(),
    //     weights.len(),
    //     weights[0].len()
    // );
    // println!("Output will be {}x{}", input.len(), weights[0].len());
    let weights_t = transpose(&weights);
    // print_a_matrix(&weights_t);
    let mut output: Vec<T> = vec![];
    if input[0].len() == weights.len() {
        for i in input.iter() {
            for j in weights_t.iter() {
                // println!("{:?}x{:?},", i, j);
                output.push(dot_product(&i, &j));
            }
        }
        // println!("{:?}", output);
        shape_changer(&output, input.len(), weights_t.len())
    } else {
        panic!("Dimension mismatch")
    }
}

pub fn dot_product<T>(a: &Vec<T>, b: &Vec<T>) -> T
where
    T: std::ops::Mul<Output = T> + std::iter::Sum + Copy,
{
    let output: T = a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum();
    output
}

pub fn element_wise_operation<T>(a: &Vec<T>, b: &Vec<T>, operation: &str) -> Vec<T>
where
    T: Copy
        + std::fmt::Debug
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::cmp::PartialEq
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    if a.len() == b.len() {
        a.iter().zip(b.iter()).map(|(x, y)| match operation {
                        "Mul" => *x * *y,
                        "Add" => *x + *y,
                        "Sub" => *x - *y,
                        "Div" => *x / *y,
                        _ => panic!("Operation unsuccessful!\nEnter any of the following(case sensitive):\n> Add\n> Sub\n> Mul\n> Div"),
                    })
                    .collect()
    } else {
        panic!("Dimension mismatch")
    }
}

pub fn pad_with_zero<T>(vector: &mut Vec<T>, count: usize) -> Vec<T>
where
    T: Copy + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let mut output = vector.clone();
    let zero = "0".parse::<T>().unwrap();
    for _ in 0..count {
        output.push(zero);
    }
    output
}

pub fn make_matrix_float<T>(input: &Vec<Vec<T>>) -> Vec<Vec<f64>>
where
    T: std::fmt::Display + Copy,
{
    println!("========================================================================================================================================================");
    input
        .iter()
        .map(|a| {
            a.iter()
                .map(|b| {
                    if is_numerical(*b) {
                        format!("{}", b).parse().unwrap()
                    } else {
                        panic!("Non numerical value present in the intput");
                    }
                })
                .collect()
        })
        .collect()
}

pub fn make_vector_float<T>(input: &Vec<T>) -> Vec<f64>
where
    T: std::fmt::Display + Copy,
{
    println!("========================================================================================================================================================");
    input
        .iter()
        .map(|b| {
            if is_numerical(*b) {
                format!("{}", b).parse().unwrap()
            } else {
                panic!("Non numerical value present in the intput");
            }
        })
        .collect()
}
pub fn round_off_f(value: f64, decimals: i32) -> f64 {
    println!("========================================================================================================================================================");
    ((value * 10.0f64.powi(decimals)).round()) / 10.0f64.powi(decimals)
}

pub fn min_max_f(list: &Vec<f64>) -> (f64, f64) {
    // println!("========================================================================================================================================================");
    if type_of(list[0]) == "f64" {
        let mut positive: Vec<f64> = list
            .clone()
            .iter()
            .filter(|a| **a >= 0.)
            .map(|a| *a)
            .collect();
        let mut negative: Vec<f64> = list
            .clone()
            .iter()
            .filter(|a| **a < 0.)
            .map(|a| *a)
            .collect();
        positive.sort_by(|a, b| a.partial_cmp(b).unwrap());
        negative.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // println!("{:?}", list);
        if negative.len() > 0 {
            (negative[0], positive[positive.len() - 1])
        } else {
            (positive[0], positive[positive.len() - 1])
        }
    } else {
        panic!("Input should be a float type");
    }
}

pub fn is_numerical<T>(value: T) -> bool {
    if type_of(&value) == "&i32"
        || type_of(&value) == "&i8"
        || type_of(&value) == "&i16"
        || type_of(&value) == "&i64"
        || type_of(&value) == "&i128"
        || type_of(&value) == "&f64"
        || type_of(&value) == "&f32"
        || type_of(&value) == "&u32"
        || type_of(&value) == "&u8"
        || type_of(&value) == "&u16"
        || type_of(&value) == "&u64"
        || type_of(&value) == "&u128"
        || type_of(&value) == "&usize"
        || type_of(&value) == "&isize"
    {
        true
    } else {
        false
    }
}

// use std::collections::HashMap;
pub fn value_counts<T>(list: &Vec<T>) -> HashMap<T, u32>
where
    T: std::cmp::PartialEq + std::cmp::Eq + std::hash::Hash + Copy,
{
    println!("========================================================================================================================================================");
    let mut count: HashMap<T, u32> = HashMap::new();
    for i in list {
        count.insert(*i, 1 + if count.contains_key(i) { count[i] } else { 0 });
    }
    count
}

use std::any::type_name;
pub fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

pub fn unique_values<T>(list: &Vec<T>) -> Vec<T>
where
    T: std::cmp::PartialEq + Copy,
{
    let mut output = vec![];
    for i in list.iter() {
        if output.contains(i) {
        } else {
            output.push(*i)
        };
    }
    output
}

pub fn element_wise_matrix_operation<T>(
    matrix1: &Vec<Vec<T>>,
    matrix2: &Vec<Vec<T>>,
    operation: &str,
) -> Vec<Vec<T>>
where
    T: Copy
        + std::fmt::Debug
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::cmp::PartialEq
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    if matrix1.len() == matrix2.len() && matrix1[0].len() == matrix2[0].len() {
        matrix1
            .iter()
            .zip(matrix2.iter())
            .map(|(x, y)| {
                x.iter()
                    .zip(y.iter())
                    .map(|a| match operation {
                        "Mul" => *a.0 * *a.1,
                        "Add" => *a.0 + *a.1,
                        "Sub" => *a.0 - *a.1,
                        "Div" => *a.0 / *a.1,
                        _ => panic!("Operation unsuccessful!\nEnter any of the following(case sensitive):\n> Add\n> Sub\n> Mul\n> Div"),
                    })
                    .collect()
            })
            .collect()
    } else {
        panic!("Dimension mismatch")
    }
}

pub fn matrix_vector_product_f(matrix: &Vec<Vec<f64>>, vector: &Vec<f64>) -> Vec<f64> {
    let mut output: Vec<_> = vec![];
    for i in matrix.iter() {
        output.push(dot_product(i, vector));
    }
    output
}
