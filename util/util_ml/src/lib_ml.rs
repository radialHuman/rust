// https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/

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
8. simple_linear_regression_prediction :
    > 1. train : A &Vec<(T,T)>
    > 2. test : A &Vec<(T,T)>
    = 1. Vec<T>
9. type_of : To know the type of a variable
    > 1. _
    = 1. str
10. unique_values : of a Vector
    > 1. list : A &Vec<T>
    = 1. Vec<T>
11. variance :
    > 1. list : A &Vec<T>
    = 1. f64
12. turn_string_categorical :
    > 1. list : A &Vec<T>
    > 2. extra_class : bool if true more than 10 classes else less
    = Vec<usize>
13. value_counts :
    > 1. list : A &Vec<T>
    = HashMap<T, u32>
14. is_numerical :
    > 1. value: T
    = bool
15. min_max_f :
    > 1. list: A &Vec<f64>
    = (f64, f64)
16. min_max :
    > 1. list: A &Vec<T>
    = (T, T)
17. normalize : between [0,1]
    > 1. list: A &Vec<f64>
    = Vec<f64>
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
pub fn read_csv(path: String, columns: i32) -> HashMap<String, Vec<String>> {
    println!("========================================================================================================================================================");
    println!("Reading the file ...");
    let file = fs::read_to_string(&path).unwrap();
    // making vec (rows)
    let x_vector: Vec<_> = file.split("\r\n").collect();
    let rows: i32 = (x_vector.len() - 1) as i32 / columns;
    println!("Input row count is {:?}", rows);
    // making vec of vec (table)
    let table: Vec<Vec<&str>> = x_vector.iter().map(|a| a.split(",").collect()).collect();
    println!("The header is {:?}", &table[0]);
    // making a dictionary
    let mut table_hashmap: HashMap<String, Vec<String>> = HashMap::new();
    for (n, i) in table[0].iter().enumerate() {
        let mut vector = vec![];
        for j in table[1..].iter() {
            vector.push(j[n]);
        }
        table_hashmap.insert(
            i.to_string(),
            vector.iter().map(|a| a.to_string()).collect(),
        );
    }
    table_hashmap
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

pub fn turn_string_categorical<T>(list: &Vec<T>, extra_class: bool) -> Vec<usize>
where
    T: std::cmp::PartialEq + std::cmp::Eq + std::hash::Hash + Copy,
{
    println!("========================================================================================================================================================");
    let values = unique_values(&list);
    if extra_class == true && values.len() > 10 {
        println!("The number of classe will be more than 10");
    } else {
        ();
    }
    let mut map: HashMap<&T, usize> = HashMap::new();
    for (n, i) in values.iter().enumerate() {
        map.insert(i, n + 1);
    }
    list.iter().map(|a| map[a]).collect()
}

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

pub fn min_max_f(list: &Vec<f64>) -> (f64, f64) {
    // check if it is a numerical type
    if type_of(list[0]) == "f64" || type_of(list[0]) == "f32" {
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
        println!("{:?}", list);
        (negative[0], positive[positive.len() - 1])
    } else {
        panic!("Input should be a float type");
    }
}

pub fn min_max<T>(list: &Vec<T>) -> (T, T)
where
    T: std::cmp::Ord + Copy,
{
    if type_of(&list[0]) == "&i32"
        || type_of(&list[0]) == "&i8"
        || type_of(&list[0]) == "&i16"
        || type_of(&list[0]) == "&i64"
        || type_of(&list[0]) == "&i128"
        || type_of(&list[0]) == "&u32"
        || type_of(&list[0]) == "&u8"
        || type_of(&list[0]) == "&u16"
        || type_of(&list[0]) == "&u64"
        || type_of(&list[0]) == "&u128"
        || type_of(&list[0]) == "&usize"
        || type_of(&list[0]) == "&isize"
    {
        (
            list.iter().map(|a| *a).min().unwrap(),
            list.iter().map(|a| *a).max().unwrap(),
        )
    } else {
        panic!("The input should be either int or unsigned")
    }
}

pub fn normalize(list: &Vec<f64>) -> Vec<f64> {
    let (minimum, maximum) = min_max(&list);
    let range: f64 = maximum - minimum;
    list.iter().map(|a| 1. - ((maximum - a) / range)).collect()
}

pub fn simple_logistic_regression_prediction() {}
