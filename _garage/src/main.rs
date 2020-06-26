use simple_ml::*;

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
    // use std::collections::BTreeMap;

    // let mut d: BTreeMap<String, Vec<i32>> = BTreeMap::new();
    // d.entry("one".to_string())
    //     .or_insert(vec![1, 2, 3, 4, 5, 6, 7, 8]);
    // d.entry("two".to_string())
    //     .or_insert(vec![5, 6, 7, 8, 8, 9, 10, 11]);
    // d.entry("three".to_string())
    //     .or_insert(vec![8, 9, 10, 11, 1, 2, 3, 4]);

    // println!("{:?}", MultivariantLinearRegression::hash_to_table(&d));
    // let x: Vec<_> = d.values().cloned().collect();
    // println!("{:?}", x[0]);

    // let mlr = MultivariantLinearRegression {
    //     header: columns,
    //     data: values,
    //     split_ratio: 0.25,
    //     alpha_learning_rate: 0.005,
    //     iterations: 1000,
    // };
    // mlr.multivariant_linear_regression();

    // let arr = vec![
    //     1., 2., 3., 4., 5., 6., 7., 8., 1., 2., 3., 4., 5., 6., 7., 8.,
    // ];
    // println!("{:?}", split_vector_at(&arr, 4.));
    // let (columns, values) = read_csv(".\util\util_ml\data\data_banknote_authentication.csv".to_string());

    // let dataset = (
    //     vec![
    //         vec![2.771244718, 1.784783929, 0.],
    //         vec![1.728571309, 1.169761413, 0.],
    //         vec![7.497545867, 3.162953546, 1.],
    //         vec![7.444542326, 0.476683375, 1.],
    //         vec![10.12493903, 3.234550982, 1.],
    //     ],
    //     vec![
    //         vec![3.678319846, 2.81281357, 0.],
    //         vec![3.961043357, 2.61995032, 0.],
    //         vec![2.999208922, 2.209014212, 0.],
    //         vec![9.00220326, 3.339047188, 1.],
    //         vec![6.642287351, 3.319983761, 1.],
    //     ],
    // );

    // let sample = (
    //     vec![vec![1.728571309, 1.169761413, 0.]],
    //     vec![
    //         vec![2.771244718, 1.784783929, 0.],
    //         vec![7.497545867, 3.162953546, 1.],
    //         vec![7.444542326, 0.476683375, 1.],
    //         vec![10.12493903, 3.234550982, 1.],
    //         vec![3.678319846, 2.81281357, 0.],
    //         vec![3.961043357, 2.61995032, 0.],
    //         vec![2.999208922, 2.209014212, 0.],
    //         vec![9.00220326, 3.339047188, 1.],
    //         vec![6.642287351, 3.319983761, 1.],
    //     ],
    // );

    // // println!("{:?}", calculate_gini_index(&sample, &vec![1., 0.]));

    // let dummy = vec![
    //     vec![2.771244718, 1.784783929, 0.],
    //     vec![1.728571309, 1.169761413, 0.],
    //     vec![7.497545867, 3.162953546, 1.],
    //     vec![7.444542326, 0.476683375, 1.],
    //     vec![10.12493903, 3.234550982, 1.],
    //     vec![3.678319846, 2.81281357, 0.],
    //     vec![3.961043357, 2.61995032, 0.],
    //     vec![2.999208922, 2.209014212, 0.],
    //     vec![9.00220326, 3.339047188, 1.],
    //     vec![6.642287351, 3.319983761, 1.],
    // ];
    // println!("{:?}", get_tree_split(&dummy));
    // println!("{:?}", to_terminal(&dummy));
    // let sm = StringToMatch {
    //     string1: String::from("Audi 3-Series"),
    //     string2: String::from("2 Series"),
    // };
    // println!(
    //     "{:?} became => {:?}",
    //     "S0meth!ng-15/hërĖ",
    //     string_to_match::clean_string("S0meth!ng-15/hërĖ".to_string())
    // );

    // println!(
    //     "Compare characters {:?}%",
    //     string_to_match::compare_chars(&sm)
    // );
    // println!(
    //     "Compare positions {:?}%",
    //     string_to_match::compare_position(&sm)
    // );

    // println!(
    //     "Compare similarity {:?}%",
    //     StringToMatch::compare_percentage(&sm, 1., 2.)
    // );

    // println!(
    //     "Compare similarity {:?}%",
    //     StringToMatch::fuzzy_subset(&sm, 3)
    // );

    // let (num, aplha) = StringToMatch::split_alpha_numericals("Something 123 else".to_string());
    // println!(
    //     "{:?} contains {:?} as numbers and {:?} as alphabets",
    //     "Something 123 else", num, aplha
    // )
    // println!(
    //     "Missing : {:?}",
    //     StringToMatch::char_count("SOmething Else".to_string())
    // );
    // println!(
    //     "Most occuring word in `Someeethings` : {:?}",
    //     StringToMatch::frequent_char("SOmething Else".to_string())
    // );
    // println!(
    //     "SOmething Else is now : {:?}",
    //     StringToMatch::char_replace("SOmething Else".to_string(), 'e', "z", "first")
    // );
    // println!(
    //     "SOmething Else is now {:?}",
    //     StringToMatch::char_replace("SOmething Else".to_string(), 'e', "Z", "all")
    // );

    // println!(
    //     "ALL : SOmething Else is now {:?}",
    //     StringToMatch::char_replace("SOmething Else".to_string(), 'e', String::from("Z"), "all")
    // );

    // println!(
    //     "First : SOmething Else is now {:?}",
    //     StringToMatch::char_replace(
    //         "SOmething Else".to_string(),
    //         'e',
    //         String::from("Z"),
    //         "first"
    //     )
    // );
    //     let string = String::from("The quick brown dog jumps Over the lazy fox");
    //     println!(
    //         "{:?}\nhas these vowels\n{:?}\nand these consonants\n{:?}",
    //         string,
    //         extract_vowels_consonants(string.clone()).0,
    //         extract_vowels_consonants(string.clone()).1
    //     );
    //     println!(
    //         "Sentence case of {:?} is\n {:?}",
    //         string.clone(),
    //         sentence_case(string.clone())
    //     );
    //     let string2 = String::from("Rust is a multi-paradigm programming language focused on performance and safety, especially safe concurrency.[15][16] Rust is syntactically similar to C++,[17] but provides memory safety without using garbage collection.
    // Rust was originally designed by Graydon Hoare at Mozilla Research, with contributions from Dave Herman, Brendan Eich, and others.[18][19] The designers refined the language while writing the Servo layout or browser engine,[20] and the Rust compiler. The compiler is free and open-source software dual-licensed under the MIT License and Apache License 2.0.");

    //     println!(
    //         "Removing stop words from\n{:?}\ngives\n{:?}",
    //         string2.clone(),
    //         remove_stop_words(string2.clone())
    //     );

    // let sample = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    // println!(
    //     "Z-score of {:?} in {:?} is {:?}",
    //     4,
    //     sample,
    //     z_score(&sample, 4)
    // );

    // println!(
    //     "{:?}",
    //     BLR_f::sigmoid_activation(&vec![
    //         0.19849274, 0.21902214, 0.68417234, 0.13458896, 0.61311337, 0.08175402, 0.52679541,
    //         0.5219286, -1.4237689, 0.98864689, 0.79766791, 0.23862188
    //     ])
    // );

    // let ts = TimeSeries {
    //     data: vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
    // };
    // println!("{:?}", ts.lag_view(3));

    // reading in data and splitting it into features and target
    // let (x, y) = BLR_f::read_n_split_n_shuffle(
    //     "data_banknote_authentication.txt",
    //     "class\r",
    // );
    // converting features and target into f64
    // let X: Vec<Vec<f64>> = x
    //     .iter()
    //     .map(|a| a.iter().map(|a| a.parse().unwrap()).collect())
    //     .collect();
    // ..1 is for removing \r in target
    // let Y: Vec<Vec<f64>> = y
    //     .iter()
    //     .map(|a| a.iter().map(|a| a[..1].parse().unwrap()).collect())
    //     .collect();

    // starting logistic regression
    // let blr = BLR_f {
    //     features: X.to_vec(),
    //     target: Y[0].clone(),
    //     learning_rate: 0.001,
    //     iterations: 500,
    // };
    // println!("{:?}", blr.features.len());

    // let train = blr.train_test_split(0.25).0;
    // let test = blr.train_test_split(0.25).1;
    // println!("{:?}", test.features[1].len());
    // println!(
    //     "Train features :{:?}\nTrain targets : {:?}\nTest features :{:?}\nTest targets : {:?}",
    //     train.features[0].len(),
    //     train.target.len(),
    //     test.features[0].len(),
    //     test.target.len()
    // );

    // let weights = blr.model_predict().0;
    // println!("The weights from entire dataset {:?}", weights);
    // let weights_train = train.model_predict().0;
    // println!("The weights from training dataset {:?}", weights_train);
    // // let weights_train = vec![-0.5248337318336532, -0.6999734220615731, 0.13801808898589332, 0.013422794479572116];
    // // &test.find_accuracy(&weights_train);
    // &test.confusion_me(&weights_train);

    // let (columns, values) = read_csv("ccpp.csv".to_string());
    // let mlr = MultivariantLinearRegression {
    //     header: columns,
    //     data: values,
    //     split_ratio: 0.25,
    //     alpha_learning_rate: 0.005,
    //     iterations: 1000,
    // };
    // mlr.multivariant_linear_regression();
    // println!();

    // let values = vec![
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "setosa",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "versicolor",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    //     "virginica",
    // ];
    // randomize_vector(&values);
    // print_a_matrix("encoded", &one_hot_encoding(&values));

    // let (columns, values) = read_csv("data_banknote_authentication.txt".to_string());
    // // let target_number = 3;

    // let column_data = row_to_columns_conversion(&make_matrix_string_literal(&values));
    // print_a_matrix("Head of column wise", &head(&column_data, 1));
    // let row_data = columns_to_rows_conversion(&column_data);
    // print_a_matrix("Head of row wise", &head(&row_data, 5));

    // let a = vec![vec![1, 2], vec![3, 5]];
    // let b = vec![vec![0, 1], vec![5, 7]];
    // print_a_matrix("Wide", &join_matrix(&a, &b, "wide"));
    // print_a_matrix("Long", &join_matrix(&a, &b, "long"));
    // let file = "../../rust/_garage/ccpp.csv".to_string();
    // let lr = OLS {
    //     file_path: file.clone(),
    //     test_size: 0.20,
    // };
    // lr.fit();

    // let m = vec![1,2,3,4,5,6,7];
    // println!("{:?}", m[..7-1].to_vec());
    // println!("{:?}", m[7..].to_vec());
    // // println!("{:?}",[&m[..4-1].to_vec()[..], &m[4..].to_vec()[..]].concat());

    // println!("\n>>>>>>>>>>>>>>>>> ORDINARY LEAST SQUARE");
    // let file = "./ccpp.csv".to_string();
    // let lr = OLS {
    //     file_path: file.clone(),
    //     target: 5,
    //     test_size: 0.20,
    // };
    // lr.fit();
    let a = vec![
        vec![1, 2, 3],
        vec![4, 5, 6],
        vec![7, 8, 9],
        vec![10, 20, 30],
        vec![40, 50, 60],
        vec![70, 80, 90],
        vec![100, 200, 300],
        vec![400, 500, 600],
        vec![700, 800, 900],
        vec![-1, -2, -3],
        vec![-4, -5, -6],
        vec![-7, -8, -9],
        vec![-10, -20, -30],
        vec![-40, -50, -60],
        vec![-70, -80, -90],
        vec![-100, -200, -300],
        vec![-400, -500, -600],
        vec![-700, -800, -900],
    ];
    println!("Train : {:?}, Test: {:?}", cv(&a, 2).0, cv(&a, 2).1);
    println!("Train : {:?}, Test: {:?}", cv(&a, 2).0, cv(&a, 2).1);
    println!("Train : {:?}, Test: {:?}", cv(&a, 2).0, cv(&a, 2).1);
}

pub fn cv<T: Copy>(data: &Vec<Vec<T>>, k: usize) -> (Vec<Vec<T>>, Vec<Vec<T>>) {
    /*
    K-fold Cross validation
    */

    (
        randomize(&data.clone())[k..].to_vec(),
        randomize(&data.clone())[..k].to_vec(),
    )
}

// ================================================================================= // =================================================================================
// ================================================================================= // =================================================================================
// ================================================================================= // =================================================================================
// ================================================================================= // =================================================================================// ================================================================================= // =================================================================================
// ================================================================================= // =================================================================================

// pub struct OLS {
//     pub file_path: String,
//     pub target: usize, // target column number
//     pub test_size: f64,
// }

// impl OLS {
//     pub fn fit(&self) {
//         /*
//         Source:
//         Video: https://www.youtube.com/watch?v=K_EH2abOp00
//         Book: Trevor Hastie,  Robert Tibshirani, Jerome Friedman - The Elements of  Statistical Learning_  Data Mining, Inference, and Pred
//         Article: https://towardsdatascience.com/regression-an-explanation-of-regression-metrics-and-what-can-go-wrong-a39a9793d914#:~:text=Root%20Mean%20Squared%20Error%3A%20RMSE,value%20predicted%20by%20the%20model.&text=Mean%20Absolute%20Error%3A%20MAE%20is,value%20predicted%20by%20the%20model.
//         Library:

//         TODO:
//         * Whats the role of gradient descent in this?
//         * rules of regression
//         * p-value
//         * Colinearity
//         */
//         // read a csv file
//         let (columns, values) = read_csv(self.file_path.clone()); // output is row wise
//                                                                   // assuming the last column has the value to be predicted
//         println!(
//             "The target here is header named: {:?}",
//             columns[self.target - 1]
//         );

//         // // converting vector of string to vector of f64s
//         let random_data = randomize(&values)
//             .iter()
//             .map(|a| {
//                 a.iter()
//                     .filter(|b| **b != "".to_string())
//                     .map(|b| b.parse::<f64>().unwrap())
//                     .collect::<Vec<f64>>()
//             })
//             .collect::<Vec<Vec<f64>>>();
//         // splitting it into train and test as per test percentage passed as parameter to get scores
//         let (train_data, test_data) = train_test_split_f(&random_data, self.test_size);
//         shape("Training data", &train_data);
//         shape("Testing data", &test_data);

//         // converting rows to vector of columns of f64s

//         // println!("{:?}",train_data );
//         shape("Training data", &train_data);
//         let actual_train = row_to_columns_conversion(&train_data);
//         // println!(">>>>>");
//         let x = drop_column(&actual_train, self.target);
//         // // the read columns are in transposed form already, so creating vector of features X and adding 1 in front of it for b0
//         let b0_vec: Vec<Vec<f64>> = vec![vec![1.; x[0].len()]]; //[1,1,1...1,1,1]
//         let X = [&b0_vec[..], &x[..]].concat(); // [1,1,1...,1,1,1]+X
//                                                 // shape(&X);
//         let xt = MatrixF { matrix: X };

//         // and vector of targets y
//         let y = vec![actual_train[self.target - 1].to_vec()];
//         // print_a_matrix(
//         //     "Features",
//         //     &xt.matrix.iter().map(|a| a[..6].to_vec()).collect(),
//         // );
//         // print_a_matrix("Target", &y);

//         /*
//         beta = np.linalg.inv(X.T@X)@(X.T@y)
//          */
//         // (X.T@X)
//         let xtx = MatrixF {
//             matrix: matrix_multiplication(&xt.matrix, &transpose(&xt.matrix)),
//         };
//         // println!("{:?}", MatrixF::inverse_f(&xtx));
//         let slopes = &matrix_multiplication(
//             &MatrixF::inverse_f(&xtx), // np.linalg.inv(X.T@X)
//             &transpose(&vec![matrix_vector_product_f(&xt.matrix, &y[0])]), //(X.T@y)
//         )[0];

//         // combining column names with coefficients
//         let output: Vec<_> = columns[..columns.len() - 1]
//             .iter()
//             .zip(slopes[1..].iter())
//             .collect();
//         // println!("****************** Without Gradient Descent ******************");
//         println!(
//         "\n\nThe coeficients of a columns as per simple linear regression on {:?}% of data is : \n{:?} and b0 is : {:?}",
//         self.test_size * 100.,
//         output,
//         slopes[0]
//     );

//         // predicting the values for test features
//         // multiplying each test feture row with corresponding slopes to predict the dependent variable
//         let mut predicted_values = vec![];
//         for i in test_data.iter() {
//             predicted_values.push({
//                 let value = i
//                     .iter()
//                     .zip(slopes[1..].iter())
//                     .map(|(a, b)| (a * b))
//                     .collect::<Vec<f64>>();
//                 value.iter().fold(slopes[0], |a, b| a + b) // b0+b1x1+b2x2..+bnxn
//             });
//         }

//         println!("RMSE : {:?}", rmse(&test_data, &predicted_values));
//         println!("MSE : {:?}", mse(&test_data, &predicted_values)); // cost function
//         println!("MAE : {:?}", mae(&test_data, &predicted_values));
//         println!("MAPE : {:?}", mape(&test_data, &predicted_values));
//         println!(
//             "R2 and adjusted R2 : {:?}",
//             r_square(
//                 &test_data
//                     .iter()
//                     .map(|a| a[test_data[0].len() - 1])
//                     .collect(), // passing only the target values
//                 &predicted_values,
//                 columns.len(),
//             )
//         );

//         println!();
//         println!();
//     }
// }

// pub fn shape(words: &str, m: &Vec<Vec<f64>>) {
//     // # of rows and columns of a matrix
//     println!("{:?} : {:?}x{:?}", words, m.len(), m[0].len());
// }

// pub fn rmse(test_data: &Vec<Vec<f64>>, predicted: &Vec<f64>) -> f64 {
//     /*
//     square root of (square of difference of predicted and actual divided by number of predications)
//     */
//     (mse(test_data, predicted)).sqrt()
// }

// pub fn mse(test_data: &Vec<Vec<f64>>, predicted: &Vec<f64>) -> f64 {
//     /*
//     square of difference of predicted and actual divided by number of predications
//     */
//     let mut square_error: Vec<f64> = vec![];
//     for (n, i) in test_data.iter().enumerate() {
//         let j = match i.last() {
//             Some(x) => (predicted[n] - x) * (predicted[n] - x), // square difference
//             _ => panic!("Something wrong in passed test data"),
//         };
//         square_error.push(j)
//     }
//     // println!("{:?}", square_error);
//     square_error.iter().fold(0., |a, b| a + b) / (predicted.len() as f64)
// }

// pub fn mae(test_data: &Vec<Vec<f64>>, predicted: &Vec<f64>) -> f64 {
//     /*
//     average of absolute difference of predicted and actual
//     */
//     let mut absolute_error: Vec<f64> = vec![];
//     for (n, i) in test_data.iter().enumerate() {
//         let j = match i.last() {
//             Some(x) => (predicted[n] - x).abs(), // absolute difference
//             _ => panic!("Something wrong in passed test data"),
//         };
//         absolute_error.push(j)
//     }
//     // println!("{:?}", absolute_error);
//     absolute_error.iter().fold(0., |a, b| a + b) / (predicted.len() as f64)
// }

// pub fn r_square(predicted: &Vec<f64>, actual: &Vec<f64>, features: usize) -> (f64, f64) {
//     // https://github.com/radialHuman/rust/blob/master/util/util_ml/src/lib_ml.rs
//     /*

//     */
//     let sst: Vec<_> = actual
//         .iter()
//         .map(|a| {
//             (a - (actual.iter().fold(0., |a, b| a + b) / (actual.len() as f64))
//                 * (a - (actual.iter().fold(0., |a, b| a + b) / (actual.len() as f64))))
//         })
//         .collect();
//     let ssr = predicted
//         .iter()
//         .zip(actual.iter())
//         .fold(0., |a, b| a + (b.0 - b.1));
//     let r2 = 1. - (ssr / (sst.iter().fold(0., |a, b| a + b)));
//     // println!("{:?}\n{:?}", predicted, actual);
//     let degree_of_freedom = predicted.len() as f64 - 1. - features as f64;
//     let ar2 = 1. - ((1. - r2) * ((predicted.len() as f64 - 1.) / degree_of_freedom));
//     (r2, ar2)
// }

// pub fn mape(test_data: &Vec<Vec<f64>>, predicted: &Vec<f64>) -> f64 {
//     /*
//     average of absolute difference of predicted and actual
//     */
//     let mut absolute_error: Vec<f64> = vec![];
//     for (n, i) in test_data.iter().enumerate() {
//         let j = match i.last() {
//             Some(x) => (((predicted[n] - x) / predicted[n]).abs()) * 100., // absolute difference
//             _ => panic!("Something wrong in passed test data"),
//         };
//         absolute_error.push(j)
//     }
//     // println!("{:?}", absolute_error);
//     absolute_error.iter().fold(0., |a, b| a + b) / (predicted.len() as f64)
// }

// pub fn drop_column(matrix: &Vec<Vec<f64>>, column_number: usize) -> Vec<Vec<f64>> {
//     // let part1 = matrix[..column_number - 1].to_vec();
//     // let part2 = matrix[column_number..].to_vec();
//     // shape("target", &part2);
//     [
//         &matrix[..column_number - 1].to_vec()[..],
//         &matrix[column_number..].to_vec()[..],
//     ]
//     .concat()
// }

// use simple_ml::*;

// pub fn row_to_columns_conversion<T: std::fmt::Debug + Copy>(data: &Vec<Vec<T>>) -> Vec<Vec<T>> {
//     /*
//     Since read_csv gives values row wise, it might be required to convert it into columns for some calulation like aggeration
//     converts [[1,6,11],[2,7,12],[3,8,13],[4,9,14],[5,10,15]] => [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]
//     */
//     println!("{:?}x{:?} becomes", data.len(), data[0].len());
//     let mut output:Vec<Vec<_>> = vec![];
//     for j in 0..(data[0].len()) {
//         let columns = data.iter().map(|a| a[j]).collect();
//         output.push(columns)
//     }
//     println!("{:?}x{:?}", output.len(), output[0].len());
//     output
// }

// ================================================================================= // =================================================================================
// ================================================================================= // =================================================================================
// ================================================================================= // =================================================================================
// ================================================================================= // =================================================================================// ================================================================================= // =================================================================================
// ================================================================================= // =================================================================================

/*
let (_, values) = read_csv("daily_minimum_temp.txt".to_string());
    let str_temp: Vec<_> = values
        .iter()
        .map(|a| (&a[1][..]).replace("\r", ""))
        .collect();
    let mut temp: Vec<f64> = str_temp.iter().map(|a| a.parse().unwrap()).collect();
    let lag: u32 = 1;
    let diff_temp = lag_n_diff_f(&mut temp, lag as usize);
    println!("{:?}\n{:?}", diff_temp.len(), diff_temp[..5].to_vec());
    let tsdall = lagmat(&diff_temp, lag);
    let trimmed_temp: Vec<Vec<f64>> = temp[1..temp.len() - 1]
        .to_vec()
        .iter()
        .map(|a| vec![*a])
        .collect();
    let mut new_tsdall = join_matrix(&tsdall, &trimmed_temp, "wide");
    let mut output = vec![];
    // tsdall[:, 0] = ts[-nobs - 1:-1]
    // tsdall[:, :maxlag + 1]
    for i in new_tsdall.iter() {
        let mut row = i.clone()[1..].to_vec();
        // row.reverse();
        output.push(row.push(1.));
    }
    let tsdshort = diff_temp[values.len() - diff_temp.len()..].to_vec();
    println!("{:?}\n{:?}", output, output.len());
    println!("{:?}\n{:?}", tsdshort, tsdshort.len());
*/

pub fn lagmat<T: Copy>(array: &Vec<T>, lag: u32) -> Vec<Vec<T>> {
    // https://gist.github.com/jcorrius/c3212b991b4f484cd502a50e7b92d41b
    let mut output = vec![];
    for (n, _) in array.iter().enumerate() {
        if n < array.len() - (lag as usize) {
            let mut subarray = array[n..(n + 1 + (lag as usize))].to_vec();
            subarray.reverse();
            output.push(subarray);
        }
    }
    output
}

pub fn lag_n_diff_f(array: &mut Vec<f64>, lag: usize) -> Vec<f64> {
    // https://gist.github.com/jcorrius/c3212b991b4f484cd502a50e7b92d41b
    let lag_temp: Vec<f64> = pad_with_zero(&mut array[lag..].to_vec(), lag, "post");
    let mut diff_temp = element_wise_operation(&lag_temp, &array, "sub");
    diff_temp = diff_temp.iter().map(|a| round_off_f(*a, 1)).collect();
    diff_temp = diff_temp[..diff_temp.len() - lag].to_vec();
    diff_temp
}

// fn find_best_split<T>(X: Vec<Vec<T>>, Y: Vec<T>) {}

// Classification tree
// https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775
// https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
// https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea

pub struct TimeSeries<T> {
    data: Vec<T>,
}
impl<T> TimeSeries<T>
where
    T: std::clone::Clone,
{
    pub fn lag_view(self, window_size: usize) -> (Vec<Vec<T>>, Vec<T>) {
        /*
            Returns sliding window of data and lagged data
        */
        // https://www.ritchievink.com/blog/2018/09/26/algorithm-breakdown-ar-ma-and-arima-models/
        let mut window_matrix = vec![];
        let reduced_data = self.data[window_size..].to_vec();
        for (n, _) in self.data[..self.data.len() - window_size]
            .iter()
            .enumerate()
        {
            window_matrix.push(self.data[n..n + window_size].to_vec());
        }
        (window_matrix, reduced_data)
    }

    pub fn ma(epsilon: Vec<T>, theta: Vec<T>) {}
}
