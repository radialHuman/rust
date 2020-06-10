fn main_() {

    // OUT OF ORDER FUNCTIONS
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
