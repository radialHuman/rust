// use simple_ml::*;

#[warn(unused_variables)]
fn main() {
    // println!(
    //     "{:?}",
    //     activation_relu(&vec![1., 3., 5., 6., 8., -3., -5., -7.])
    // );

    // println!("Numerical {:?}", is_numerical(23.67));

    // let mut v = vec![1., 3., 5., 6., 8., -3., -5., -7.];
    // let (min, max) = min_max_float(&v);
    // println!("min is {:?}\nmax is {:?}", min, max);

    // println!("{:?}\nNormalized to \n{:?}", v, normalize(&v));

    // let v1 = vec![
    //     5, 6, 8, 4, 2, 3, 5, 9, 7, 4, 1, 2, 35, 6, 45, 48, 4, 21, 6, 13, 2168, 1, 5, 68, 1, -45, 0,
    // ];
    // let (min, max) = min_max(&v1);
    // println!("min is {:?}\nmax is {:?}", min, max);

    // let a = vec![vec![1, 2], vec![2, 3], vec![3, 7], vec![34, 76]];
    // let b = vec![vec![0.2, 0.3], vec![0.4, 0.7], vec![1., 2.], vec![0.6, 0.]];
    // println!("{:?}", logit_function_f(&a, &b));
    // println!("{:?} becomes\n{:?}", &v1, make_vector_float(&v1));
    // let v2 = vec![1.356785, 2.56836, 5.807422];
    // let rv: Vec<_> = v2.iter().map(|a| round_off_f(*a, 3)).collect();
    // println!("{:?}\n{:?}", v2, rv);

    // let a_f = vec![vec![1., 2.], vec![2., 3.], vec![3., 7.], vec![4., 6.]];
    // let mut b = vec![vec![0.2, 0.3], vec![0.4, 0.7], vec![1., 2.], vec![0.6, 1.]];
    // let y = vec![
    //     vec![5., 6., 1., 2.],
    //     vec![5., 87., 1., 2.],
    //     vec![8., 2., 1., 2.],
    //     vec![1., 1., 1., 2.],
    // ];
    // println!("{:?}", y.len());
    // println!("{:?}", cost_function(&a_f, &b, &y));
    // println!("{:?}", matrix_subtraction(&a_f, &b));

    // let log = vec![
    //     vec![-3.71100666, -1.52977611, -6.71534849, -7.16446920],
    //     vec![-2.41008454, -5.35627762, -3.35406373, -1.48842547],
    //     vec![-6.50435618, -2.24035625, -4.13993765, -1.50721716],
    //     vec![-7.16446920, -3.02298093, -1.12535168, -2.24842045],
    // ];
    // println!("{:?}", cost_function(&a_f, &b, &y));
    // println!("Mul :\n{:?}", element_wise_operation(&v2, &v2, "Mul"));
    // println!("Add :\n{:?}", element_wise_operation(&v2, &v2, "Add"));
    // println!("Sub :\n{:?}", element_wise_operation(&v2, &v2, "Sub"));
    // println!("Div :\n{:?}", element_wise_operation(&v2, &v2, "Div"));
    // println!(
    //     "Something else :\n{:?}",
    //     element_wise_operation(&v2, &v1, "Mul")
    // );

    // println!(
    //     "{:?}x{:?}",
    //     &matrix_multiplication(&log, &a_f).len(),
    //     &matrix_multiplication(&log, &a_f)[0].len()
    // );
    // println!(
    //     "{:?}",
    //     shape_changer(&matrix_multiplication(&log, &a_f), &matrix_multiplication(&log, &a_f).len(), 4)
    // );
    // println!("{:?}", transpose(&matrix_multiplication(&log, &a_f)));
    // println!("{:?}", gradient_descent(&a_f, &mut b, &y, 0.01, 0.001));
    // let target = vec![1., 5., 3., 4.];
    // println!("{:?}", matrix_vector_product_f(&a_f, &target));
    // println!("{:?}", logistic_regression("C:\Users\rahul.damani\Downloads\code\rust\util\util_ml\data\dataset_iris.txt", "species", 0.25));
    //
    // let list = vec![
    //     vec![
    //         0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4,
    //         0.3, 0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2,
    //         0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 1.4,
    //         1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0, 1.3, 1.4, 1.0, 1.5, 1.0, 1.4, 1.3, 1.4, 1.5, 1.0,
    //         1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7, 1.5, 1.0,
    //     ],
    //     vec![
    //         1.1, 1.0, 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3, 1.2, 1.4, 1.2, 1.0, 1.3, 1.2, 1.3,
    //         1.3, 1.1, 1.3, 2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8, 1.8, 2.5, 2.0, 1.9, 2.1, 2.0,
    //         2.4, 2.3, 1.8, 2.2, 2.3, 1.5, 2.3, 2.0, 2.0, 1.8, 2.1, 1.8, 1.8, 1.8, 2.1, 1.6, 1.9,
    //         2.0, 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9, 2.3, 2.5, 2.3, 1.9, 2.0,
    //         2.3, 1.8,
    //     ],
    // ];

    // let x = vec![vec![1., 2.], vec![2., 3.], vec![3., 7.], vec![4., 6.]];
    // let mut b = vec![vec![0.2, 0.3], vec![0.4, 0.7], vec![1., 2.], vec![0.6, 1.]];
    // let y = vec![1., 5., 3., 7.];
    // println!("{:?}", log_gradient_f(&x, &b, &y));
    // println!("{:?}", gradient_descent(&x, &mut b, &y, 0.01, 0.001));
    // println!("{:?}", logistic_predict(&a_f, &b));
    // println!("{:?}", logistic_function_f(&a_f, &b));
    // read_csv(
    //     "..//util//util_ml//data\\data_banknote_authentication.txt".to_string(),
    //     5,
    // );

    // let (beta, iteration_count, features, target) =
    // binary_logistic_regression(
    //     "..//util//util_ml//data\\data_banknote_authentication.txt".to_string(),
    //     "class".to_string(),
    //     0.25,
    //     0.01,
    //     0.001,
    // );
    // println!("The coefficients are:\n{:?}", beta);
    // println!("{:?}",logistic_predict(&features, &beta));
    // let dict: HashMap<&str, Vec<i32>> = [
    //     ("one", vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    //     ("two", vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    //     ("three", vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    //     ("four", vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    // ]
    // .iter()
    // .cloned()
    // .collect();
    // // println!("{:?}", randomize_before_split(&dict));
    // println!("{:?}", train_test_split(&dict, 0.25));

    // println!(
    //     "{:?}",
    //     randomize_before_split(vec![
    //         "1", "2", "4", "7", "5", "3", "2", "7", "7", "9", "0", "2", "6", "32", "76", "97",
    //         "43",
    //     ])
    // );

    // let m_2 = MatrixF {
    //     matrix: vec![vec![1., 2.], vec![2., 3.]],
    // };
    // let m_3 = MatrixF {
    //     matrix: vec![vec![1., 2., 4.], vec![2., 3., 1.], vec![2., 3., 5.]],
    // };
    // let log = MatrixF {
    //     matrix: vec![
    //         vec![
    //             -4.310025518872387,
    //             -4.214165016957441,
    //             -7.062973356056997,
    //             -0.06913842034334694,
    //         ],
    //         vec![
    //             -5.141851064900488,
    //             -86.05215356307842,
    //             -1.0022378485212764,
    //             -0.0030184163247083395,
    //         ],
    //         vec![
    //             -0.006692850924284732,
    //             -0.00033535013046637197,
    //             -0.000000041399375594330934,
    //             -0.00000011253516207787584,
    //         ],
    //         vec![
    //             -1.0691384203433467,
    //             -1.014774031693273,
    //             -1.0001507103580596,
    //             -1.0002248167702334,
    //         ],
    //     ],
    // };
    // println!("{:?}", m_2.inverse_f());
    // println!("{:?}", m_3.inverse_f());
    // println!("{:?}", log.inverse_f());

    // let straight = MatrixF {
    //     matrix: vec![
    //         vec![5., 4., 3., 2., 1.],
    //         vec![4., 3., 2., 1., 5.],
    //         vec![3., 2., 9., 5., 4.],
    //         vec![2., 1., 5., 4., 3.],
    //         vec![1., 2., 3., 4., 5.],
    //     ],
    // };
    // print_a_matrix(
    //     &format!("And the inverse of \n{:?}\n is", &straight),
    //     &straight
    //         .inverse_f()
    //         .iter()
    //         .map(|a| a.iter().map(|b| round_off_f(*b, 3)).collect())
    //         .collect(),
    // );
    // https://medium.com/@pytholabs/multivariate-linear-regression-from-scratch-in-python-5c4f219be6a
    // https://medium.com/we-are-orb/multivariate-linear-regression-in-python-without-scikit-learn-7091b1d45905
    let (columns, values) = read_csv("ccpp.csv".to_string());
    // println!("{:?}", columns);
    // println!("{:?}", values);

    // println!(
    //     "{:?}",
    //     mse_cost_function(
    //         &vec![vec![1., 2.], vec![2., 3.], vec![3., 7.], vec![4., 6.]],
    //         &vec![1., 5., 3., 7.],
    //         &vec![0.1, 0.2],
    //     )
    // );

    //

    let mlr = MultivariantLinearRegression {
        header: columns,
        data: values,
        split_ratio: 0.25,
        alpha_learning_rate: 0.005,
        iterations: 100,
    };
    mlr.multivariant_linear_regression();
    // let (a, b) = MultivariantLinearRegression::batch_gradient_descent(
    //     &vec![vec![1., 2.], vec![2., 3.], vec![3., 7.], vec![4., 6.]],
    //     &vec![1., 5., 3., 7.],
    //     &vec![0.1, 0.2],
    //     0.01,
    //     100,
    // );
    // println!("{:?}\n{:?}", a, b);
    // let loss = element_wise_operation(
    //     &matrix_vector_product_f(
    //         &vec![vec![1., 2.], vec![2., 3.], vec![3., 7.], vec![4., 6.]],
    //         &vec![0.1, 0.2],
    //     ),
    //     &vec![1., 5., 3., 7.],
    //     "Sub",
    // );
    // let features = &vec![vec![1., 2.], vec![2., 3.], vec![3., 7.], vec![4., 6.]];
    // let theta = vec![0.1, 0.2];
    // let gradient: Vec<_> = matrix_vector_product_f(&transpose(&features), &loss)
    //     .iter()
    //     .map(|a| (a / vec![1., 5., 3., 7.].len() as f64) * 0.01)
    //     .collect();
    // let new_theta = element_wise_operation(&theta, &gradient, "Sub").clone();
    // println!("{:?}", new_theta)

    let mut d: HashMap<String, Vec<i32>> = HashMap::new();
    d.entry("one".to_string()).or_insert(vec![1, 2, 3, 4]);
    // d.entry("two".to_string()).or_insert(vec![5, 6, 7, 8]);
    // d.entry("three".to_string()).or_insert(vec![8, 9, 10, 11]);

    // println!("{:?}", MultivariantLinearRegression::hash_to_table(&d));
    let x: Vec<i32> = d.values().into_iter();
    println!("{:?}", x);
}

struct MultivariantLinearRegression {
    header: Vec<String>,
    data: Vec<Vec<String>>,
    split_ratio: f64,
    alpha_learning_rate: f64,
    iterations: i32,
}

impl MultivariantLinearRegression {
    pub fn multivariant_linear_regression(&self) -> () {
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
        let mut train_feature = HashMap::new();
        let mut test_feature = HashMap::new();
        let mut train_target = HashMap::new();
        let mut test_target = HashMap::new();
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
        let mut norm_test_features = HashMap::new();
        let mut norm_train_features = HashMap::new();
        let mut norm_test_target = HashMap::new();
        let mut norm_train_target = HashMap::new();
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

        coefficients = vec![0.; train.len()];
        // MultivariantLinearRegression::batch_gradient_descent(
        //     &MultivariantLinearRegression::hash_to_table(&norm_train_features),
        //     &norm_test_features.values().collect(),
        //     &coefficients,
        //     self.alpha_learning_rate,
        //     self.iterations,
        // );
    }

    fn mse_cost_function(features: &Vec<Vec<f64>>, target: &Vec<f64>, theta: &Vec<f64>) -> f64 {
        let rows = target.len();
        let numerator: Vec<_> =
            element_wise_operation(&matrix_vector_product_f(features, theta), target, "Sub")
                .iter()
                .map(|a| *a * *a)
                .collect();
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
            // println!("theta:  {:?}", new_theta);
        }
        (new_theta.clone(), cost_history)
    }

    pub fn hash_to_table<T: Copy>(d: &HashMap<String, Vec<T>>) -> Vec<Vec<T>> {
        // changes the order of table columns
        let mut vector = vec![];
        for (_, v) in d.iter() {
            vector.push(v.clone());
        }
        let mut original = vec![];
        for i in 0..vector.len() + 1 {
            let mut row = vec![];
            for j in vector.iter() {
                row.push(j[i]);
            }
            original.push(row);
        }
        original
    }
}

// // ================================================================================================================================================
// // ================================================================================================================================================
// // ================================================================================================================================================
fn zero_matrix(row: usize, columns: usize) -> Vec<Vec<f64>> {
    let mut output: Vec<Vec<f64>> = vec![];
    for _ in 0..row {
        output.push(vec![0.; columns]);
    }
    output
}

pub fn matrix_vector_product_f(matrix: &Vec<Vec<f64>>, vector: &Vec<f64>) -> Vec<f64> {
    let mut output: Vec<_> = vec![];
    for i in matrix.iter() {
        output.push(dot_product(i, vector));
    }
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

pub fn dot_product<T>(a: &Vec<T>, b: &Vec<T>) -> T
where
    T: std::ops::Mul<Output = T> + std::iter::Sum + Copy,
{
    let output: T = a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum();
    output
}

use std::any::type_name;
pub fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
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

pub fn normalize_vector_f(list: &Vec<f64>) -> Vec<f64> {
    // println!("========================================================================================================================================================");
    let (minimum, maximum) = min_max_f(&list);
    let range: f64 = maximum - minimum;
    list.iter().map(|a| 1. - ((maximum - a) / range)).collect()
}

use std::collections::HashMap;
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

use std::fs;
pub fn read_csv<'a>(path: String) -> (Vec<String>, Vec<Vec<String>>) {
    println!("========================================================================================================================================================");
    println!("Reading the file ...");
    let file = fs::read_to_string(&path).unwrap();
    let splitted: Vec<&str> = file.split("\n").collect();
    let rows: i32 = (splitted.len() - 1) as i32;
    println!("Number of rows = ~{}", rows - 1);
    let table: Vec<Vec<_>> = splitted.iter().map(|a| a.split(",").collect()).collect();
    let values = table[1..]
        .iter()
        .map(|a| a.iter().map(|b| b.to_string()).collect())
        .collect();
    let columns: Vec<String> = table[0].iter().map(|a| a.to_string()).collect();
    (columns, values)
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
