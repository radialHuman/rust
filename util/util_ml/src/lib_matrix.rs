/*
DESCRIPTION
-----------------------------------------
STRUCTS
-------
1. MatrixF : matrix: Vec<Vec<f64>> // upto 100x100
    > determinant_f
    > inverse_f
    > is_square_matrix
    > round_off_f

2. DataFrame : string: Vec<Vec<&'a str>>, numbers: Vec<Vec<f64>>, boolean: Vec<Vec<bool>>,
    > groupby : string_column_number , operation: &str // "sum" or "mean"
    > describe

2. DataMap : string: HashMap<&'a str,Vec<&'a str>>, numbers:HashMap<&'a str,Vec<f64>>, boolean: HashMap<&'a str,Vec<bool>>,
    > groupby : string_column_number , operation: &str // "sum" or "mean"
    > describe

FUNCTIONS
---------
1. dot_product :
    > 1. A &Vec<T>
    > 2. A &Vec<T>
    = 1. T

2. element_wise_operation : for vector
    > 1. A &mut Vec<T>
    > 2. A &mut Vec<T>
    > 3. operation &str ("add","sub","mul","div")
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
    > 3. fucntion : &str ("add","sub","mul","div")
    = A Vec<Vec<T>>

18. matrix_vector_product_f
    > 1. matrix: &Vec<Vec<f64>>
    > 2. vector: &Vec<f64>
    = Vec<f64>

19. split_vector
    > 1. vector: &Vec<T>
    > 2. parts: i32
     = Vec<Vec<T>>

20. split_vector_at
    > 1. vector: &Vec<T>
    > 2. at: T
     = Vec<Vec<T>>

21. join_matrix
    > 1. matrix1: &Vec<Vec<T>>
    > 2. matrix2: &Vec<Vec<T>>
    > 3. how: &str : "long" or "wide"
    = Vec<Vec<T>> 

22. make_matrix_string_literal
    > 1. data: &'a Vec<Vec<String>>
    = Vec<Vec<&'a str>> 

23. head
    > 1. data: &Vec<Vec<T>>
    > 2. rows: usize
    = Vec<Vec<T>> 

24. tail
    > 1. data: &Vec<Vec<T>>
    > 2. rows: usize
    = Vec<Vec<T>> 

25. row_to_columns_conversion
    > 1. data: &Vec<Vec<T>>
    = Vec<Vec<T>> 

26. columns_to_rows_conversion
    > 1. data: &Vec<Vec<T>>
    = Vec<Vec<T>> 
*/

#[derive(Debug)] // to make it usable by print!
pub struct MatrixF {
    matrix: Vec<Vec<f64>>,
}

impl MatrixF {
    pub fn determinant_f(&self) -> f64 {
        // https://integratedmlai.com/find-the-determinant-of-a-matrix-with-pure-python-without-numpy-or-scipy/
        // check if it is a square matrix
        if MatrixF::is_square_matrix(&self.matrix) == true {
            println!("Calculating Determinant...");

            match self.matrix.len() {
                1 => self.matrix[0][0],
                2 => MatrixF::determinant_2(&self),
                3..=100 => MatrixF::determinant_3plus(&self),
                _ => {
                    println!("Cant find determinant for size more than {}", 100);
                    "0".parse().unwrap()
                }
            }
        } else {
            panic!("The input should be a square matrix");
        }
    }
    fn determinant_2(&self) -> f64 {
        (self.matrix[0][0] * self.matrix[1][1]) - (self.matrix[1][0] * self.matrix[1][0])
    }

    fn determinant_3plus(&self) -> f64 {
        // converting to upper triangle and multiplying the diagonals
        let length = self.matrix.len() - 1;
        let mut new_matrix = self.matrix.clone();

        // rounding off value
        new_matrix = new_matrix
            .iter()
            .map(|a| a.iter().map(|a| MatrixF::round_off_f(*a, 3)).collect())
            .collect();

        for diagonal in 0..=length {
            for i in diagonal + 1..=length {
                if new_matrix[diagonal][diagonal] == 0.0 {
                    new_matrix[diagonal][diagonal] = 0.001;
                }
                let scalar = new_matrix[i][diagonal] / new_matrix[diagonal][diagonal];
                for j in 0..=length {
                    new_matrix[i][j] = new_matrix[i][j] - (scalar * new_matrix[diagonal][j]);
                }
            }
        }
        let mut product = 1.;
        for i in 0..=length {
            product *= new_matrix[i][i]
        }
        product
    }

    pub fn is_square_matrix<T>(matrix: &Vec<Vec<T>>) -> bool {
        if matrix.len() == matrix[0].len() {
            true
        } else {
            false
        }
    }

    pub fn round_off_f(value: f64, decimals: i32) -> f64 {
        // println!("========================================================================================================================================================");
        ((value * 10.0f64.powi(decimals)).round()) / 10.0f64.powi(decimals)
    }

    pub fn inverse_f(&self) -> Vec<Vec<f64>> {
        // https://integratedmlai.com/matrixinverse/
        let mut input = self.matrix.clone();
        let length = self.matrix.len();
        let mut identity = MatrixF::identity_matrix(length);

        let mut index: Vec<usize> = (0..length).collect();
        let mut int_index: Vec<i32> = index.iter().map(|a| *a as i32).collect();

        for diagonal in 0..length {
            let diagonalScalar = 1. / (input[diagonal][diagonal]);
            // first action
            for columnLoop in 0..length {
                input[diagonal][columnLoop] *= diagonalScalar;
                identity[diagonal][columnLoop] *= diagonalScalar;
            }

            // second action
            let mut exceptDiagonal: Vec<usize> = index[0..diagonal]
                .iter()
                .copied()
                .chain(index[diagonal + 1..].iter().copied())
                .collect();
            println!("Here\n{:?}", exceptDiagonal);

            for i in exceptDiagonal {
                let rowScalar = input[i as usize][diagonal].clone();
                for j in 0..length {
                    input[i][j] = input[i][j] - (rowScalar * input[diagonal][j]);
                    identity[i][j] = identity[i][j] - (rowScalar * identity[diagonal][j])
                }
            }
        }

        identity
    }

    fn identity_matrix(size: usize) -> Vec<Vec<f64>> {
        let mut output: Vec<Vec<f64>> = MatrixF::zero_matrix(size, size);
        for i in 0..=(size - 1) {
            for j in 0..=(size - 1) {
                if i == j {
                    output[i][j] = 1.;
                } else {
                    output[i][j] = 0.;
                }
            }
        }
        output
    }

    fn zero_matrix(row: usize, columns: usize) -> Vec<Vec<f64>> {
        let mut output: Vec<Vec<f64>> = vec![];
        for _ in 0..row {
            output.push(vec![0.; columns]);
        }
        output
    }
}


pub struct DataFrame<'a> {
    // stored column wise
    pub string: Vec<Vec<&'a str>>,
    pub numerical: Vec<Vec<f64>>,
    pub boolean: Vec<Vec<bool>>,
}
impl<'a> DataFrame<'a> {
    pub fn describe(&self) {
        println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        println!("      Details of the DataFrame", );
        println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        for (n, i) in self.string.iter().enumerate() {
            println!(
                "String column #{:?}  Values count : {:?}",
                n,
                value_counts(i)
            )
        }
        for (n, i) in self.boolean.iter().enumerate() {
            println!(
                "String column #{:?}  Values count : {:?}",
                n,
                value_counts(i)
            )
        }
        for (n, i) in self.numerical.iter().enumerate() {
            println!("Numerical column #{:?}\n\tCount :{:?}", n, i.len());
            println!(
                "\tMinimum :{:?}  Maximum : {:?}",
                min_max_f(i).0,
                min_max_f(i).1
            );
            println!("\tMean :{:?}  Std Deviation : {:?}", mean(i), std_dev(i));
            quartile_f(i);
            println!("\tOutliers :{:?}", z_outlier_f(i));
        }
    }
    pub fn groupby(&self, string_column_number: usize, operation: &str) {
        // removing other string columns and boolean columns as they dont play any role

        let reduced_dataframe_string = self.string[string_column_number].clone();
        let reduced_dataframe_float = self.numerical.clone();

        // finding unique string values and the index they occur in
        let unique_string = unique_values(&reduced_dataframe_string);
        let mut unique_string_index = vec![];
        for i in unique_string.iter() {
            let mut single_string = vec![];
            for (n, j) in reduced_dataframe_string.iter().enumerate() {
                if i == j {
                    single_string.push(n);
                }
            }
            unique_string_index.push(single_string);
        }

        // operating on numerical columns
        let mut output = vec![];
        for i in unique_string_index.iter() {
            let mut result = vec![];
                for j in reduced_dataframe_float.iter() {
                        let seperated = j.iter().enumerate().filter(|(n,_)| i.contains(n) ).collect::<Vec<(usize,&f64)>>();
                        match operation {
            "sum" => {
                result.push(seperated.iter().map(|a| a.1).fold(0.,|a,b| a+b));
            }
            "mean" => {
                result.push(seperated.iter().map(|a| a.1).fold(0.,|a,b| a+b)/(seperated.len() as f64));
            }
            _ => panic!("Enter either 'sum' or 'mean'"),
        };
                    }
                    output.push(result[0]);
                }
        println!("Grouped on {:?} => {:?}",string_column_number ,unique_string.iter().zip(output.iter()).collect::<Vec<(&&str,&f64)>>());
    }
}


pub struct DataMap<'a> {
    // use std::collections::HashMap;
    // stored column wise
    pub string: HashMap<&'a str,Vec<&'a str>>,
    pub numerical: HashMap<&'a str,Vec<f64>>,
    pub boolean: HashMap<&'a str,Vec<bool>>,
}
impl<'a> DataMap<'a> {
    pub fn describe(&self) {
        println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        println!("      Details of the DataMap", );
        println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        for (k,v) in self.string.iter() {
            println!(
                "String column :{:?}  Values count : {:?}",
                k,
                value_counts(v)
            )
        }
        for (k,v) in self.boolean.iter() {
            println!(
                "Boolean column :{:?}  Values count : {:?}",
                k,
                value_counts(v)
            )
        }
        for (k,v) in self.numerical.iter() {
            println!("Numerical column :{:?}\n\tCount :{:?}", k, v.len());
            println!(
                "\tMinimum :{:?}  Maximum : {:?}",
                min_max_f(v).0,
                min_max_f(v).1
            );
            println!("\tMean :{:?}  Std Deviation : {:?}", mean(v), std_dev(v));
            quartile_f(v);
            println!("\tOutliers :{:?}", z_outlier_f(v));
        }
    }

    pub fn groupby(&self, string_column: &str, operation: &str) {
        // removing other string columns and boolean columns as they dont play any role

        let reduced_dataframe_string = self.string[string_column].clone();
        let reduced_dataframe_float:Vec<&Vec<f64>> = self.numerical.values().clone().collect();

        // finding unique string values and the index they occur in
        let unique_string = unique_values(&reduced_dataframe_string);
        let mut unique_string_index = vec![];
        for i in unique_string.iter() {
            let mut single_string = vec![];
            for (n, j) in reduced_dataframe_string.iter().enumerate() {
                if i == j {
                    single_string.push(n);
                }
            }
            unique_string_index.push(single_string);
        }

        // operating on numerical columns
        let mut output = vec![];
        for i in unique_string_index.iter() {
            let mut result = vec![];
            for j in reduced_dataframe_float.iter() {
                let seperated = j
                    .iter()
                    .enumerate()
                    .filter(|(n, _)| i.contains(n))
                    .collect::<Vec<(usize, &f64)>>();
                match operation {
                    "sum" => {
                        result.push(seperated.iter().map(|a| a.1).fold(0., |a, b| a + b));
                    }
                    "mean" => {
                        result.push(
                            seperated.iter().map(|a| a.1).fold(0., |a, b| a + b)
                                / (seperated.len() as f64),
                        );
                    }
                    _ => panic!("Enter either 'sum' or 'mean'"),
                };
            }
            output.push(result[0]);
        }
        println!(
            "Grouped by {:?} => {:?}",string_column,
            unique_string
                .iter()
                .zip(output.iter())
                .collect::<Vec<(&&str, &f64)>>()
        );
    }
}


use std::collections::HashMap;
pub struct DataMap<'a> {
    // stored column wise
    string: HashMap<&'a str,Vec<&'a str>>,
    numerical: HashMap<&'a str,Vec<f64>>,
    boolean: HashMap<&'a str,Vec<bool>>,
}
impl<'a> DataMap<'a> {
    pub fn describe(&self) {
        println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        println!("      Details of the DataMap", );
        println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        for (k,v) in self.string.iter() {
            println!(
                "String column :{:?}  Values count : {:?}",
                k,
                value_counts(v)
            )
        }
        for (k,v) in self.boolean.iter() {
            println!(
                "Boolean column :{:?}  Values count : {:?}",
                k,
                value_counts(v)
            )
        }
        for (k,v) in self.numerical.iter() {
            println!("Numerical column :{:?}\n\tCount :{:?}", k, v.len());
            println!(
                "\tMinimum :{:?}  Maximum : {:?}",
                min_max_f(v).0,
                min_max_f(v).1
            );
            println!("\tMean :{:?}  Std Deviation : {:?}", mean(v), std_dev(v));
            quartile_f(v);
            println!("\tOutliers :{:?}", z_outlier_f(v));
        }
    }

    pub fn groupby(&self, string_column: &str, operation: &str) {
        // removing other string columns and boolean columns as they dont play any role

        let reduced_dataframe_string = self.string[string_column].clone();
        let reduced_dataframe_float:Vec<&Vec<f64>> = self.numerical.values().clone().collect();

        // finding unique string values and the index they occur in
        let unique_string = unique_values(&reduced_dataframe_string);
        let mut unique_string_index = vec![];
        for i in unique_string.iter() {
            let mut single_string = vec![];
            for (n, j) in reduced_dataframe_string.iter().enumerate() {
                if i == j {
                    single_string.push(n);
                }
            }
            unique_string_index.push(single_string);
        }

        // operating on numerical columns
        let mut output = vec![];
        for i in unique_string_index.iter() {
            let mut result = vec![];
            for j in reduced_dataframe_float.iter() {
                let seperated = j
                    .iter()
                    .enumerate()
                    .filter(|(n, _)| i.contains(n))
                    .collect::<Vec<(usize, &f64)>>();
                match operation {
                    "sum" => {
                        result.push(seperated.iter().map(|a| a.1).fold(0., |a, b| a + b));
                    }
                    "mean" => {
                        result.push(
                            seperated.iter().map(|a| a.1).fold(0., |a, b| a + b)
                                / (seperated.len() as f64),
                        );
                    }
                    _ => panic!("Enter either 'sum' or 'mean'"),
                };
            }
            output.push(result[0]);
        }
        println!(
            "{:?}",
            unique_string
                .iter()
                .zip(output.iter())
                .collect::<Vec<(&&str, &f64)>>()
        );
    }
}


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
    /*
    Returns a tuple with mininmum and maximum value in a vector
    */
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
        if negative.len() > 0 && positive.len() >0
        {
            (negative[0], positive[positive.len() - 1])
        }
        else{
           if positive.len()==0 && negative.len()!=0 
                {
                 (negative[negative.len() - 1], negative[0])
                } 
            else {
            if negative.len()==0 && positive.len()!=0 
            {
             (positive[0],positive[positive.len() - 1])
            }
            else{
                panic!("Empty vector found")
            }
        }}
    }
    else 
    {  
    panic!("Input should be a float type")
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

use std::collections::HashMap;
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

pub fn split_vector<T: std::clone::Clone>(vector: &Vec<T>, parts: i32) -> Vec<Vec<T>> {
    if vector.len() % parts as usize == 0 {
        let mut output = vec![];
        let size = vector.len() / parts as usize;
        let mut from = 0;
        let mut to = from + size;
        while to <= vector.len() {
            output.push(vector[from..to].to_vec());
            from = from + size;
            to = from + size;
        }
        output
    } else {
        panic!("This partition is not possible, check the number of partiotions passed")
    }
}

pub fn split_vector_at<T>(vector: &Vec<T>, at: T) -> Vec<Vec<T>>
where
    T: std::cmp::PartialEq + Copy + std::clone::Clone,
{
    if vector.contains(&at) {
        let mut output = vec![];
        let copy = vector.clone();
        let mut from = 0;
        for (n, i) in vector.iter().enumerate() {
            if i == &at {
                output.push(copy[from..n].to_vec());
                from = n;
            }
        }
        output.push(copy[from..].to_vec());
        output
    } else {
        panic!("The value is not in the vector, please check");
    }
}


pub fn join_matrix<T: Copy>(
    matrix1: &Vec<Vec<T>>,
    matrix2: &Vec<Vec<T>>,
    how: &str,
) -> Vec<Vec<T>> {
    /*
    "wide" : Places matrix next to each other to become one wide matrix
    "long" : Places matrix one below other to become a longer matrix
    */
    let mut output = vec![];
    let mut a = matrix1;
    let mut b = matrix2;
    match how {
        "wide" => {
            /*
            [[1,2],[3,5]] join_matrix [[0,1],[5,7]] => [[1,2,0,1],[3,5,5,7]]
            */
            if a.len() == b.len() {
                for (n, j) in a.iter().enumerate() {
                    let mut new_j = j.clone();
                    for (m, i) in b.iter().enumerate() {
                        for k in i.iter() {
                            if n == m {
                                new_j.push(*k);
                            }
                        }
                    }
                    output.push(new_j)
                }
                output
            } else {
                panic!("Please check the dimensions, # of rows are different");
            }
        }
        "long" => {
            /*
            [[1,2],[3,5]] join_matrix [[0,1],[5,7]] => [[1,2],[3,5],[0,1],[5,7]]
            */
            if a[0].len() == b[0].len() {
                for (n, _) in b.iter().enumerate() {
                    output.push(a[n].clone());
                }
                for (n, _) in b.iter().enumerate() {
                    output.push(b[n].clone());
                }
                output
            } else {
                panic!("Please check the dimensions, # of columns are different");
            }
        }
        _ => panic!("Select either long or wide"),
    }
}

pub fn make_matrix_string_literal<'a>(data: &'a Vec<Vec<String>>) -> Vec<Vec<&'a str>> {
    /*
    Few Copy does not work on String so convert it to &str
    */
    let mut output = vec![];
    for i in data.iter() {
        output.push(i.iter().map(|a| &a[..]).collect())
    }
    println!("> String converted to &str");
    output
}


pub fn head<T: std::clone::Clone+std::fmt::Debug>(data: &Vec<Vec<T>>, rows: usize) {
    /*
    Works on row wise data
    Shows first few rows of a matrix
    */
    if rows <= data.len() {
        let output = data[..rows].to_vec();
        print_a_matrix(&format!("First {} rows",rows), &output);
    } else {
        panic!("Data is nt that big, please check the numbers");
    }
}

pub fn tail<T: std::clone::Clone+std::fmt::Debug>(data: &Vec<Vec<T>>, rows: usize) {
    /*
    Works on row wise data
    Shows first few rows of a matrix
    */
    if rows <= data.len() {
        let output = data[data.len()-rows..].to_vec();
        print_a_matrix(&format!("Last {} rows",rows), &output);
    } else {
        panic!("Data is nt that big, please check the numbers");
    }
}

pub fn row_to_columns_conversion<T: std::fmt::Debug + Copy>(data: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    /*
    Since read_csv gives values row wise, it might be required to convert it into columns for some calulation like aggeration
    converts [[1,6,11],[2,7,12],[3,8,13],[4,9,14],[5,10,15]] => [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]
    */
    // println!("{:?}x{:?} becomes", data.len(), data[0].len());
    let mut output = vec![];
    for j in 0..data[0].len() {
        let mut columns = vec![];
        for i in data.iter() {
            columns.push(i[j]);
        }
        output.push(columns)
    }
    // println!("{:?}x{:?}", output.len(), output[0].len());
    output
}

pub fn columns_to_rows_conversion<T: std::fmt::Debug + Copy>(data: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    /*
    Opposite of row_to_columns_conversion
    converts  [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]] => [[1,6,11],[2,7,12],[3,8,13],[4,9,14],[5,10,15]]
    */
    // println!("{:?}x{:?} becomes", data.len(), data[0].len());
    let mut output = vec![];
    for j in 0..data[0].len() {
        let mut columns = vec![];
        for i in data.iter() {
            columns.push(i[j]);
        }
        output.push(columns)
    }
    // println!("{:?}x{:?}", output.len(), output[0].len());
    output
}