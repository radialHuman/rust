/*
DESCRIPTION
-----------------------------------------
STRUCTS
-------
1. MultivariantLinearRegression : header: Vec<String>, data: Vec<Vec<String>>, split_ratio: f64, alpha_learning_rate: f64, iterations: i32,
> multivariant_linear_regression
> batch_gradient_descent
> hash_to_table
x generate_score
x mse_cost_function
x train_test_split
x randomize

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

16. logistic_predict :
    1. > matrix1: &Vec<Vec<f64>>
    2. > beta: &Vec<Vec<f64>>
    = Vec<Vec<f64>>

17. randomize :
    1. > rows : &Vec<f64>
    = Vec<f64>

18. train_test_split :
    1. > input: &Vec<f64>
    2. > percentage: f64
    = Vec<f64>
    = Vec<f64>

19. binary_logistic_regression :
    1. path: String
    2. target_name: String
    3. test_percentage: f64
    4. learning_rate : f64
    5. coverage_rate : f64
    = beta : Vec<Vec<f64>>
    = # of iterations : i32

*/

use crate::lib_matrix;
use lib_matrix::*;

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
    let splitted: Vec<&str> = file.split("\n").collect();
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

pub fn randomize(rows: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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

pub fn train_test_split(input: &Vec<Vec<f64>>, percentage: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
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

// pub fn binary_logistic_regression(
//     path: String,
//     target_name: String,
//     test_percentage: f64,
//     learning_rate: f64,
//     coverage_rate: f64,
// ) -> (Vec<Vec<f64>>, i32) {
//     // use std::collections::HashMap;
//     let (columns, values) = read_csv(path);
//     // converting input to str and normalizing them
//     let mut df: HashMap<String, Vec<f64>> = HashMap::new();
//     for (n, i) in columns.iter().enumerate() {
//         let mut v = vec![];
//         for j in values.iter() {
//             for (m, k) in j.iter().enumerate() {
//                 if n == m {
//                     v.push(k.parse().unwrap());
//                 }
//             }
//         }
//         v = normalize_vector_f(&v);
//         df.insert(i.to_string(), v);
//     }
//     // print!("{:?}", df);
//     // test and train split, target and features split
//     let mut test_features: HashMap<String, Vec<f64>> = HashMap::new();
//     let mut train_features: HashMap<String, Vec<f64>> = HashMap::new();
//     let mut test_target: HashMap<String, Vec<f64>> = HashMap::new();
//     let mut train_target: HashMap<String, Vec<f64>> = HashMap::new();

//     for (k, v) in df.iter() {
//         if *k.to_string() != target_name {
//             test_features.insert(k.clone(), train_test_split(v, test_percentage).1);
//             train_features.insert(k.clone(), train_test_split(v, test_percentage).0);
//         // X
//         } else {
//             test_target.insert(k.clone(), train_test_split(v, test_percentage).1);
//             train_target.insert(k.clone(), train_test_split(v, test_percentage).0);
//             // y
//         }
//     }
//     let feature_vector: Vec<_> = train_features.values().cloned().collect();
//     let target_vector: Vec<_> = train_target.values().cloned().collect();
//     let feature_length = feature_vector[0].len();
//     // println!("{:?}", target_vector);

//     // initiating beta values
//     let mut beta_df = HashMap::new();
//     for (n, i) in columns.iter().enumerate() {
//         let mut v = vec![0.; feature_length];
//         beta_df.insert(i.to_string(), v);
//     }

//     let mut beta = vec![vec![0.; train_features.keys().len()]];
//     println!("BETA: {:?}", beta);

//     // gradient descent on beta
//     let (new_beta, iteration_count) =
//         gradient_descent(&feature_vector, &mut beta, &target_vector[0], 0.01, 0.001);
//     // println!(
//     //     "{:?}\n{:?}\n{:?}\n{:?}\n{:?}",
//     //     feature_vector, target_vector, &beta, &new_beta, iteration_count
//     // );
//     (new_beta, iteration_count)
// }

pub struct MultivariantLinearRegression {
    pub header: Vec<String>,
    pub data: Vec<Vec<String>>,
    pub split_ratio: f64,
    pub alpha_learning_rate: f64,
    pub iterations: i32,
}

use std::collections::BTreeMap;
impl MultivariantLinearRegression {
    //
    // https://medium.com/we-are-orb/multivariate-linear-regression-in-python-without-scikit-learn-7091b1d45905
    pub fn multivariant_linear_regression(&self)
    //-> (Vec<f64>, Vec<f64>)
    {
        // removing incomplete data
        println!(
            "Before removing missing values, number of rows : {:?}",
            self.data.len()
        );
        let df_na_removed: Vec<_> = self
            .data
            .iter()
            .filter(|a| a.len() == self.header.len())
            .collect();
        println!(
            "After removing missing values, number of rows : {:?}",
            df_na_removed.len()
        );
        // assuming the last column has the value to be predicted
        println!(
            "The target here is header named: {:?}",
            self.header[self.header.len() - 1]
        );

        // converting values to floats
        let df_f: Vec<Vec<f64>> = df_na_removed
            .iter()
            .map(|a| a.iter().map(|b| b.parse::<f64>().unwrap()).collect())
            .collect();
        println!("Values are now converted to f64");

        // shuffling splitting test and train
        let (train, test) = train_test_split(&df_f, self.split_ratio);
        println!("Train size: {}\nTest size : {:?}", train.len(), test.len());

        // feature and target split
        let mut train_feature = BTreeMap::new();
        let mut test_feature = BTreeMap::new();
        let mut train_target = BTreeMap::new();
        let mut test_target = BTreeMap::new();
        let mut coefficients = vec![];

        // creating training dictionary
        for (n, j) in self.header.iter().enumerate() {
            if *j != self.header[self.header.len() - 1] {
                let mut row = vec![];
                for i in train.iter() {
                    row.push(i[n]);
                }
                train_feature.entry(j.to_string()).or_insert(row);
            } else {
                let mut row = vec![];
                for i in train.iter() {
                    row.push(i[n]);
                }
                train_target.entry(j.to_string()).or_insert(row);
            }
        }
        // creating training dictionary
        for (n, j) in self.header.iter().enumerate() {
            if *j != self.header[self.header.len() - 1] {
                {
                    let mut row = vec![];
                    for i in test.iter() {
                        row.push(i[n]);
                    }
                    test_feature.entry(j.to_string()).or_insert(row);
                }
            } else {
                let mut row = vec![];
                for i in test.iter() {
                    row.push(i[n]);
                }
                test_target.entry(j.to_string()).or_insert(row);
            }
        }

        // normalizing values
        let mut norm_test_features = BTreeMap::new();
        let mut norm_train_features = BTreeMap::new();
        let mut norm_test_target = BTreeMap::new();
        let mut norm_train_target = BTreeMap::new();
        for (k, _) in test_feature.iter() {
            norm_test_features
                .entry(k.clone())
                .or_insert(normalize_vector_f(&test_feature[k]));
        }
        for (k, _) in train_feature.iter() {
            norm_train_features
                .entry(k.clone())
                .or_insert(normalize_vector_f(&train_feature[k]));
        }
        for (k, _) in test_target.iter() {
            norm_test_target
                .entry(k.clone())
                .or_insert(normalize_vector_f(&test_target[k]));
        }
        for (k, _) in train_target.iter() {
            norm_train_target
                .entry(k.clone())
                .or_insert(normalize_vector_f(&train_target[k]));
        }
        // println!("{:?}", norm_test_target);

        coefficients = vec![0.; train[0].len() - 1];
        let target: Vec<_> = norm_train_target.values().cloned().collect();
        // println!("TARGET\n{:?}", target[0].len());
        let (coefficeints, _) = MultivariantLinearRegression::batch_gradient_descent(
            &MultivariantLinearRegression::hash_to_table(&norm_train_features),
            &target[0],
            &coefficients,
            self.alpha_learning_rate,
            self.iterations,
        );
        println!("The weights of the inputs are {:?}", coefficeints);
        let mut pv: Vec<_> = MultivariantLinearRegression::hash_to_table(&norm_test_features)
            .iter()
            .map(|a| element_wise_operation(a, &coefficeints, "Mul"))
            .collect();

        let mut predicted_values = vec![];
        for i in pv.iter() {
            predicted_values.push(i.iter().fold(0., |a, b| a + b))
        }

        let a = &MultivariantLinearRegression::hash_to_table(&norm_test_target);
        let mut actual = vec![];
        for i in a.iter() {
            actual.push(i[0]);
        }

        println!(
            "The r2 of this model is : {:?}",
            MultivariantLinearRegression::generate_score(&predicted_values, &actual)
        );
    }

    fn generate_score(predicted: &Vec<f64>, actual: &Vec<f64>) -> f64 {
        let sst: Vec<_> = actual
            .iter()
            .map(|a| {
                (a - (actual.iter().fold(0., |a, b| a + b) / (actual.len() as f64))
                    * (a - (actual.iter().fold(0., |a, b| a + b) / (actual.len() as f64))))
            })
            .collect();
        let ssr = predicted
            .iter()
            .zip(actual.iter())
            .fold(0., |a, b| a + (b.0 - b.1));
        let r2 = 1. - (ssr / (sst.iter().fold(0., |a, b| a + b)));
        // println!("{:?}\n{:?}", predicted, actual);
        r2
    }

    fn mse_cost_function(features: &Vec<Vec<f64>>, target: &Vec<f64>, theta: &Vec<f64>) -> f64 {
        let rows = target.len();
        let prod = matrix_vector_product_f(&features, theta);
        // println!(">>>>>>>>\n{:?}x{:?}", prod.len(), target.len(),);
        let numerator: Vec<_> = element_wise_operation(&prod, target, "Sub")
            .iter()
            .map(|a| *a * *a)
            .collect();
        // print!(".");
        numerator.iter().fold(0., |a, b| a + b) / (2. * rows as f64)
    }

    pub fn batch_gradient_descent(
        features: &Vec<Vec<f64>>,
        target: &Vec<f64>,
        theta: &Vec<f64>,
        alpha_lr: f64,
        max_iter: i32,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut new_theta = theta.clone();
        let mut hypothesis_value = vec![];
        let mut cost_history = vec![];
        let mut loss = vec![];
        let mut gradient = vec![];
        let rows = target.len();
        for _ in 0..max_iter {
            hypothesis_value = matrix_vector_product_f(features, &new_theta);
            loss = hypothesis_value
                .iter()
                .zip(target)
                .map(|(a, b)| a - b)
                .collect();

            gradient = matrix_vector_product_f(&transpose(features), &loss)
                .iter()
                .map(|a| a / rows as f64)
                .collect();

            new_theta = element_wise_operation(
                &new_theta,
                &gradient.iter().map(|a| alpha_lr * a).collect(),
                "Sub",
            )
            .clone();

            cost_history.push(MultivariantLinearRegression::mse_cost_function(
                features, target, &new_theta,
            ));
        }
        println!("");
        (new_theta.clone(), cost_history)
    }

    fn train_test_split(input: &Vec<Vec<f64>>, percentage: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // shuffle
        let data = MultivariantLinearRegression::randomize(input);
        // println!("{:?}", data);
        // split
        let test_count = (data.len() as f64 * percentage) as usize;
        // println!("Test size is {:?}", test_count);

        let test = data[0..test_count].to_vec();
        let train = data[test_count..].to_vec();
        (train, test)
    }

    fn randomize(rows: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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

    pub fn hash_to_table<T: Copy + std::fmt::Debug>(d: &BTreeMap<String, Vec<T>>) -> Vec<Vec<T>> {
        // changes the order of table columns
        let mut vector = vec![];
        for (_, v) in d.iter() {
            vector.push(v.clone());
        }
        let mut original = vec![];
        for i in 0..vector[0].len() {
            let mut row = vec![];
            for j in vector.iter() {
                row.push(j[i]);
            }
            original.push(row);
        }
        original
    }
}

// // decision tree
// pub struct TreeClassifier {
//     pub header: Vec<String>,
//     pub data: Vec<Vec<String>>,
//     pub split_ratio: f64,
// }

// impl TreeClassifier {
//     //
//     // https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

// fn get_tree_split(data: &Vec<Vec<f64>>) -> (usize, f64, (Vec<Vec<f64>>, Vec<Vec<f64>>)) {
//     // data is row by row without test or train or feature or target split
//     // assuming the labels are in the end of each row
//     let class_value: Vec<_> = data.iter().map(|a| a[a.len() - 1]).collect();
//     println!("The labels are {:?}", unique_values(&class_value));
//     let (mut b_index, mut b_value, mut b_score) = (999, 999., 999.);
//     let mut b_groups = (vec![], vec![]);
//     for i in 0..data[0].len() - 1 {
//         for j in data {
//             let groups = train_test_split(i, j[i], data);
//             println!("{:?}", class_value);
//             let gini = calculate_gini_index(&groups, &class_value); // --> error in calculation
//             println!("***** {:?}:{:?}", j[i], gini);
//             if gini < b_score {
//                 b_index = i;
//                 b_value = j[i];
//                 b_score = gini;
//                 b_groups = groups;
//             }
//         }
//     }
//     (b_index, b_value, b_groups)
// }

// fn calculate_gini_index(group: &(Vec<Vec<f64>>, Vec<Vec<f64>>), class: &Vec<f64>) -> f64 {
//     let instances = group.0.len() as f64 + group.1.len() as f64;
//     let mut gini = 0.;
//     let g = vec![group.0.clone(), group.1.clone()];
//     for i in g {
//         let size = i.len() as f64;
//         if size != 0. {
//             let mut score = 0.;
//             for j in class {
//                 let mut p = vec![];
//                 for k in i.clone() {
//                     p.push(k[k.len() - 1]);
//                 }
//                 let count: Vec<_> = p.iter().filter(|a| *a == j).collect();
//                 let p_score = count.len() as f64 / size;
//                 score += p_score as f64 * p_score as f64;
//             }
//             gini += (1. - score) * (size / instances);
//         }
//     }
//     gini
// }

// fn train_test_split(
//     index: usize,
//     value: f64,
//     data: &Vec<Vec<f64>>,
// ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
//     let (mut left, mut right) = (vec![], vec![]);
//     for row in data {
//         if row[index] < value {
//             left.push(row.clone());
//         } else {
//             right.push(row.clone())
//         }
//     }
//     // println!("{:?}\n{:?}", left, right);
//     return (left, right);
// }

// use std::collections::HashMap;
// fn to_terminal(group: &Vec<Vec<f64>>) -> f64 {
//     // value count of classes
//     let v: Vec<_> = group.iter().map(|a| a[a.len() - 1]).collect();
//     let mut count: HashMap<String, u32> = HashMap::new();
//     for i in v {
//         count.insert(
//             i.to_string(),
//             1 + if count.contains_key(&i.to_string()) {
//                 count[&i.to_string()]
//             } else {
//                 0
//             },
//         );
//     }
//     // which class occurs the most
//     let mut max_value: u32 = 0;
//     let mut max_key = String::new();
//     for (k, v) in count.iter() {
//         if *v > max_value {
//             max_value = *v;
//             max_key = k.to_string();
//         }
//     }
//     max_key.parse().unwrap()
// }

// fn split(node: &Vec<Vec<f64>>, max_depth: i32, min_size: i32, depth: i32) {
//     let mut output = HashMap::new();
//     let (mut left, mut right) = (node[0].clone(), node[1].clone());
//     // if there are no values, combine and send it to to_terminal
//     if
//     output["left"]
// }
// }
