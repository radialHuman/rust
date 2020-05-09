use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::io;

use std::any::type_name;

fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

#[derive(Debug)]
pub struct Columns<T> {
    name: String,
    values: Vec<T>,
}

// the table
#[derive(Debug)]
pub struct Table {
    symboling: i32,
    normalized_losses: i32,
    make: String,
    fuel_type: String,
    aspiration: String,
    num_of_doors: String,
    body_style: String,
    drive_wheels: String,
    engine_location: String,
    wheel_base: f64,
    length: f64,
    width: f64,
    height: f64,
    curb_weight: u32,
    engine_type: String,
    num_of_cylinders: String,
    engine_size: u32,
    fuel_system: String,
    bore: f64,
    stroke: f64,
    compression_ratio: f64,
    horsepower: u32,
    peak_rpm: u32,
    city_mpg: u32,
    highway_mpg: u32,
    price: u32,
}

fn main() {
    let columns_count = 26;
    let rows_count = 206;

    // reading the file as string
    let x: String = read_file().unwrap();
    println!("File starting with '{:?}', successfully read", &x[..10]);
    let csv = x.replace("\r\n", ",").replace("?", "9999999"); // changing new line characters to commas and missing value ? to 0 as it is only in a numerical column

    // splitting csv
    let data: Vec<_> = csv.split(",").collect();
    println!("Total cells are {}", data.len());
    // println!("{:?}", &data);

    // remove ""
    let clean_data: Vec<_> = data.iter().filter(|x| **x != "").collect();
    println!("Length of clean data is {:?}", clean_data.len());

    // making it a table
    let df = shape_changer(&clean_data, columns_count, rows_count);
    print_a_table(&df, 5);

    // columns are
    println!("Columns are \n{:?}", &df[0]);

    // creating a new table for change in types
    let mut row_wise_new_df: Vec<Table> = vec![];

    // assigning data types to each column
    // println!("{:?}", String::from(*df[1][0]).parse::<i32>().unwrap());
    for rows in 1..df.len() {
        row_wise_new_df.push(Table {
            symboling: String::from(*df[rows][0]).parse::<i32>().unwrap(),
            normalized_losses: String::from(*df[rows][1]).parse::<i32>().unwrap(),
            make: String::from(*df[rows][2]).to_string(),
            fuel_type: String::from(*df[rows][3]).to_string(),
            aspiration: String::from(*df[rows][4]).to_string(),
            num_of_doors: String::from(*df[rows][5]).to_string(),
            body_style: String::from(*df[rows][6]).to_string(),
            drive_wheels: String::from(*df[rows][7]).to_string(),
            engine_location: String::from(*df[rows][8]).to_string(),
            wheel_base: String::from(*df[rows][9]).parse::<f64>().unwrap(),
            length: String::from(*df[rows][10]).parse::<f64>().unwrap(),
            width: String::from(*df[rows][11]).parse::<f64>().unwrap(),
            height: String::from(*df[rows][12]).parse::<f64>().unwrap(),
            curb_weight: String::from(*df[rows][13]).parse::<u32>().unwrap(),
            engine_type: String::from(*df[rows][14]).to_string(),
            num_of_cylinders: String::from(*df[rows][15]).to_string(),
            engine_size: String::from(*df[rows][16]).parse::<u32>().unwrap(),
            fuel_system: String::from(*df[rows][17]).to_string(),
            bore: String::from(*df[rows][18]).parse::<f64>().unwrap(),
            stroke: String::from(*df[rows][19]).parse::<f64>().unwrap(),
            compression_ratio: String::from(*df[rows][20]).parse::<f64>().unwrap(),
            horsepower: String::from(*df[rows][21]).parse::<u32>().unwrap(),
            peak_rpm: String::from(*df[rows][22]).parse::<u32>().unwrap(),
            city_mpg: String::from(*df[rows][23]).parse::<u32>().unwrap(),
            highway_mpg: String::from(*df[rows][24]).parse::<u32>().unwrap(),
            price: String::from(*df[rows][25]).parse::<u32>().unwrap(),
        });
    }

    // make column wise df
    let cwndf = transpose(&df);
    let mut column_wise_new_df: Vec<_> = vec![];

    for i in cwndf.iter() {
        let column = Columns {
            name: String::from(i[0].replace("-", "_").clone()),
            values: i
                .iter()
                .filter(|x| **x != i[0])
                .map(|z| String::from(**z))
                .collect(),
        };
        column_wise_new_df.push(column);
    }
    // statistics
    find_missing(&column_wise_new_df, "num_of_doors");
    unique_values(&column_wise_new_df, "num_of_doors");
    find_average(&column_wise_new_df, "bore");
}

pub fn find_average(table: &Vec<Columns<String>>, column_name: &str) {
    let mut average = 0.;
    let mut length = 0;
    for i in table.iter() {
        if i.name == column_name.to_string() {
            if type_of(i.values[0].clone()) != "String" {
                average = i
                    .values
                    .iter()
                    .map(|z| z.parse().unwrap())
                    .filter(|z| z != &9999999.)
                    .collect::<Vec<f64>>()
                    .iter()
                    .sum::<f64>();
                length = i.values.iter().len();
            }
        }
        // else {
        //     println!("This column cant be averaged");
        // }
    }
    average = average / (length as f64);
    println!("\n\nAverage of {} is {}", column_name, average);
}

pub fn unique_values(table: &Vec<Columns<String>>, column_name: &str) {
    let mut unique = vec![];
    for i in table.iter() {
        if i.name == column_name.to_string() {
            for j in i.values.iter() {
                if unique.contains(j) {
                    "";
                } else {
                    unique.push(j.to_string());
                }
            }
        }
    }
    println!("\n\nUnique value in {} are {:?}", column_name, unique);
}

pub fn find_missing(table: &Vec<Columns<String>>, column_name: &str) {
    let mut missing_count = 0;
    for i in table.iter() {
        if i.name == column_name.to_string() {
            let x: Vec<_> = i.values.iter().filter(|x| x == &"9999999").collect();
            missing_count = x.len();
        }
    }
    println!("\n\n{} Missing value in {}", missing_count, column_name);
}

pub fn read_file() -> Result<String, io::Error> {
    fs::read_to_string("C:/Users/rahul.damani/Downloads/code/rust/eda_1/src/Automobile_data.csv")
}

pub fn shape_changer<T>(list: &Vec<T>, columns: usize, rows: usize) -> Vec<Vec<T>>
where
    T: std::clone::Clone,
{
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

pub fn print_a_table<T>(matrix: &Vec<Vec<T>>, row_count: usize)
where
    T: std::fmt::Debug,
{
    println!("\n\nThe head of the table is:");
    for i in matrix[..row_count].iter() {
        println!("{:?}", i);
    }
    println!("");
    println!("");
}

pub fn transpose<T>(matrix: &Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: std::clone::Clone,
{
    let mut output = vec![];
    for j in 0..matrix[0].len() {
        for i in 0..matrix.len() {
            output.push(matrix[i][j].clone());
        }
    }
    let x = matrix[0].len();
    shape_changer(&output, matrix.len(), x)
}

/*
 OUTPUT
 File starting with '"symboling,"', successfully read
Total cells are 5357
Length of clean data is 5356


The head of the table is:
["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
["3", "9999999", "alfa-romero", "gas", "std", "two", "convertible", "rwd", "front", "88.6", "168.8", "64.1", "48.8", "2548", "dohc", "four", "130", "mpfi", "3.47", "2.68", "9", "111", "5000", "21", "27", "13495"]
["3", "9999999", "alfa-romero", "gas", "std", "two", "convertible", "rwd", "front", "88.6", "168.8", "64.1", "48.8", "2548", "dohc", "four", "130", "mpfi", "3.47", "2.68", "9", "111", "5000", "21", "27", "16500"]
["1", "9999999", "alfa-romero", "gas", "std", "two", "hatchback", "rwd", "front", "94.5", "171.2", "65.5", "52.4", "2823", "ohcv", "six", "152", "mpfi", "2.68", "3.47", "9", "154", "5000", "19", "26", "16500"]
["2", "164", "audi", "gas", "std", "four", "sedan", "fwd", "front", "99.8", "176.6", "66.2", "54.3", "2337", "ohc", "four", "109", "mpfi", "3.19", "3.4", "10", "102", "5500", "24", "30", "13950"]


Columns are
["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]


2 Missing value in num_of_doors


Unique value in num_of_doors are ["two", "four", "9999999"]


Average of bore is 3.264780487804879
 */
