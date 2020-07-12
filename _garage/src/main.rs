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
    // let a = vec![
    //     vec![1, 2, 3],
    //     vec![4, 5, 6],
    //     vec![7, 8, 9],
    //     vec![10, 20, 30],
    //     vec![40, 50, 60],
    //     vec![70, 80, 90],
    //     vec![100, 200, 300],
    //     vec![400, 500, 600],
    //     vec![700, 800, 900],
    //     vec![-1, -2, -3],
    //     vec![-4, -5, -6],
    //     vec![-7, -8, -9],
    //     vec![-10, -20, -30],
    //     vec![-40, -50, -60],
    //     vec![-70, -80, -90],
    //     vec![-100, -200, -300],
    //     vec![-400, -500, -600],
    //     vec![-700, -800, -900],
    // ];
    // println!("Train : {:?}, Test: {:?}", cv(&a, 2).0, cv(&a, 2).1);
    // println!("Train : {:?}, Test: {:?}", cv(&a, 2).0, cv(&a, 2).1);
    // println!("Train : {:?}, Test: {:?}", cv(&a, 2).0, cv(&a, 2).1);

    // let v: Vec<f64> = vec![
    //     -1000000.0, -100000.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
    //     14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0,
    //     29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0,
    //     44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0,
    //     59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0,
    //     74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0,
    //     89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 120.0, 200.0,
    //     652.0, 1258.0, 5123.0, 6542.0, 452846.0, 500345.0,
    // ];
    // println!("Outliers {:?}", z_outlier_f(&v));
    // quartile_f(&v);
    // println!("27th percentile: {:?}", percentile_f(&v,27));

    // let df = DataFrame {
    //     string: vec![vec![
    //         "One", "Two", "One", "Two", "One", "One", "Two", "Two", "One", "One", "Two", "One",
    //         "One","-One", "-Two", "-One", "-Two", "-One", "-One", "-Two", "-Two", "-One", "-One", "-Two", "-One",
    //         "-One",
    //     ],
    //     vec!["Positive","Positive","Positive","Positive","Positive","Positive","Positive","Positive","Positive","Positive","Positive","Positive","Positive",
    //     "Negative","Negative","Negative","Negative","Negative","Negative","Negative","Negative","Negative","Negative","Negative","Negative","Negative"]
    //     ],
    //     numbers: vec![
    //         vec![1., 2., 1., 2., 1., 1., 2., 2., 1., 1., 2., 1., 1., -1., -2., -1., -2., -1., -1., -2., -2., -1., -1., -2., -1., -1.],
    //         // vec![1., 2., 1., 2., 1., 1., 2., 2., 1., 1., 2., 1., 1., -1., -2., -1., -2., -1., -1., -2., -2., -1., -1., -2., -1., -1.],
    //         // vec![1., 2., 1., 2., 1., 1., 2., 2., 1., 1., 2., 1., 1., -1., -2., -1., -2., -1., -1., -2., -2., -1., -1., -2., -1., -1.],
    //     ],
    //     boolean: vec![vec![]],
    // };
    // df.groupby(0, "sum");
    // df.groupby(0, "mean");
    // df.groupby(1, "sum");
    // df.groupby(1, "mean");

    // let s = "The plus sign, +, is a binary operator that indicates addition, as in 2 + 3 = 5. It can also serve as a unary operator that leaves its operand unchanged (+x means the same as x). This notation may be used when it is desired to emphasize the positiveness of a number, especially when contrasting with the negative (+5 versus −5).";
    // println!(
    //     "{:?} =>\n{:?}",
    //     s,
    //     tokenize(s.to_string(), &vec!["+", ")", "("])
    // );

    // let df = DataFrame {
    //     string: vec![
    //         vec![
    //             "One", "Two", "Three", "One", "Two", "Three", "One", "Two", "Three", "One", "Two",
    //             "Three",
    //         ],
    //         vec!["1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3"],
    //     ],
    //     numerical: vec![
    //         vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 12., 11.],
    //         vec![
    //             -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12.,
    //         ],
    //     ],
    //     boolean: vec![vec![
    //         true, false, true, true, true, false, true, true, true, false, true, true,
    //     ]],
    // };
    // df.describe();
    // df.groupby(1,"sum");

    // // creating hashmaps
    // let mut string_columns:HashMap<&str,Vec<&str>> = HashMap::new();
    // string_columns.insert("string_1",vec!["One", "Two", "Three", "One", "Two", "Three", "One", "Two", "Three", "One", "Two","Three"]);
    // string_columns.insert("string_2",vec!["1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3"]);
    // let mut numerical_columns:HashMap<&str,Vec<f64>> = HashMap::new();
    // numerical_columns.insert("numerical_1",vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 12., 11.]);
    // numerical_columns.insert("numerical_2",vec![-1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12.,]);
    // let mut boolean_columns:HashMap<&str,Vec<bool>> = HashMap::new();
    // boolean_columns.insert("boolean_1",vec![true, false, true, true, true, false, true, true, true, false, true, true]);

    // let dm = DataMap {
    //     string:  string_columns,
    //     numerical: numerical_columns,
    //     boolean: boolean_columns,};
    // dm.describe();
    // dm.groupby("string_2", "mean");

    // let mut ts = vec![
    //     -213., -564., -35., -15., 141., 115., -420., -360., 203., -338., -431., 194., -220., -513.,
    //     154., -125., -559., 92., -21., -579., -52., 99., -543., -175., 162., -457., -346., 204.,
    //     -300., -474., 164., -107., -572., -8., 83., -541., -224., 180., -420., -374., 201., -236.,
    //     -531., 83., 27., -564., -112., 131., -507., -254., 199., -311., -495., 143., -46., -579.,
    //     -90., 136., -472., -338., 202., -287., -477., 169., -124., -568., 17., 48., -568., -135.,
    //     162., -430., -422., 172., -74., -577., -13., 92., -534., -243., 194., -355., -465., 156.,
    //     -81., -578., -64., 139., -449., -384., 193., -198., -538., 110., -44., -577., -6., 66.,
    //     -552., -164., 161., -460., -344., 205., -281., -504., 134., -28., -576., -118., 156.,
    //     -437., -381., 200., -220., -540., 83., 11., -568., -160., 172., -414., -408., 188., -125.,
    //     -572., -32., 139., -492., -321., 205., -262., -504., 142., -83., -574., 0., 48., -571.,
    //     -106., 137., -501., -266., 190., -391., -406., 194., -186., -553., 83., -13., -577., -49.,
    //     103., -515., -280., 201., 300., -506., 131., -45., -578., -80., 138., -462., -361., 201.,
    //     -211., -554., 32., 74., -533., -235., 187., -372., -442., 182., -147., -566., 25., 68.,
    //     -535., -244., 194., -351., -463., 174., -125., -570., 15., 72., -550., -190., 172., -424.,
    //     -385., 198., -218., -536., 96.,
    // ];

    // for i in 0..10 {
    //     println!(
    //         "Lag: {:?}\tAutocorrelation: {:?}",
    //         i,
    //         round_off_f(autocorrelation(&ts, i).unwrap(), 3)
    //     );
    // }

    // let mut ts = vec![
    //     1.5, 0.9, -0.1, -0.3, -0.7, 0.2, -1.0, -0.2, -1.1, -1.0, -0.8, -0.7, -0.9, -0.6, 0.5, -0.2,
    //     0.8, 0.7, 0.5, 0.1, -0.2, 0.4, 0.0, -1.2, 0.3, -0.5, -0.2, 0.2, -1.7, 0.1, -0.0, -1.2,
    //     -1.2, 0.1, -0.3, -0.5, 0.7, -0.3, 0.3, 0.6, 0.5, 0.1, 0.4, 1.1, 0.2, 0.3, 0.1, 1.4, -0.5,
    //     1.9, 0.6, -0.1, 1.0, 1.3, 1.6, 1.5, 1.3, 1.5, 1.2, 1.0, 1.3, 1.6, 1.3, 0.9, 1.4, 1.1, 1.1,
    //     1.2, 0.1, 1.8, 0.2, 1.1, 0.6, -0.1, 0.2, 0.2, -0.7, 0.2, 0.2, -0.5, -0.9, 0.0, -1.0, -0.3,
    //     -1.9, -0.5, 0.3, 0.4, -0.5, 0.3, 0.4, 0.9, -0.3, 0.1, -0.4, -0.6, -0.9, -1.4, 1.3, 0.4,
    //     0.5, -0.1, -0.3, -0.1, 0.0, 0.5, 0.9, 0.9, 0.1, 0.1, 1.0, 0.8, 0.5, 0.1, 0.5, 0.8, 0.7,
    //     0.1, 0.5, 0.8, -0.3, 0.9, -1.8, 0.8, 0.3, 0.1, 0.2, 0.2, 0.1, -0.3, 0.5, 1.5, 2.0, -0.3,
    //     0.1, 0.2, 1.1, 0.7, 0.1, 0.6, 0.4, 1.0, 0.3, 0.2, 1.0, 0.6, 1.1, 0.8, 0.4, -0.5, -0.1, 0.0,
    //     -0.6, -1.2, -0.8, -1.2, -0.4, 0.0, 1.1, 1.1, 0.2, 0.8, 0.6, 1.5, 1.3, 1.3, 0.2, -0.3, -0.4,
    //     0.4, 0.8, -0.5, 0.2, -0.6, -1.8, -0.7, -1.3, -0.9, -1.5, 0.2, -1.3, -0.2, -0.9, -0.2, -0.4,
    //     0.3, 0.1, 0.6, -0.2, -0.1, -0.0, -0.3, 1.7, 1.7, 1.2, -0.0, -0.0, 0.6, 0.2, 0.7, 0.5, 0.1,
    //     -0.4, -0.6, 0.5, 1.3, 0.1, 0.0, 1.2, 1.1, 0.7, 0.3, -0.3, -0.1, -0.3, 0.2, -1.5, -0.5, 0.4,
    //     -0.4, -0.2, 0.2, -0.5, -0.1, -1.0, -0.9, -0.1, 0.4, -1.1, -1.0, 0.6, -0.1, 0.4, 1.0, -0.4,
    //     0.6, 1.2, 1.0, 1.6, 1.9, 0.4, 2.0, 2.3, 1.7, 0.9, 0.4, 0.2, 1.5, 1.4, 1.7, 0.5, 0.3, 0.5,
    //     1.1, 0.6, -0.1, -1.6, -0.5, -1.4, -0.5, -1.4, -0.9, -0.3, -1.3, -0.3, -1.4, 0.7, 0.1, 0.4,
    //     1.1, 0.6, 1.3, 1.2, 0.8, 2.6, 1.8, 2.4, 2.1, 2.4, 0.9, 1.0, 0.4, 1.1, 0.8, 1.1, 0.9, -0.1,
    //     0.2, -0.5, 0.8, 1.6, 1.2, 0.6, 1.3, 1.9, 2.0, 2.1, 1.7, 1.2, 2.1, 1.4, 1.7, 1.7, 0.3, 0.4,
    //     0.3, -0.9, -0.8, -1.1, 0.4, 0.7, -0.3, -0.4, 1.2, -0.5, -0.4, -0.5, -0.9, -1.1, -0.8, -0.8,
    //     -1.0, -1.3, -1.1, -0.3, -2.7, -0.2, -0.3, 1.4, 0.7, 0.6, 0.4, 0.1, -0.4, -0.0, -0.1, -0.4,
    //     1.0, 0.1, -0.1, 0.5, 0.6, 1.3, -0.2, 0.1, -0.1, -0.4, -0.7, 0.4, 0.3, -0.4, -0.6, -0.0,
    //     -0.0, -0.0, 0.4, -0.5, 0.0, -0.3, -0.1, 0.4, -0.2, -1.3, -0.1, 0.8, 1.2, 0.8, 0.7, 0.3,
    //     -0.4, 0.2, 0.5, 1.2, 0.3, 0.6, 0.1, -0.3, -1.0, -1.5, -1.7, -1.7, -1.7, -2.6, -2.2, -3.5,
    //     -3.6, -2.4, -0.9, 0.6, 0.0, -0.6, 0.1, 0.9, 0.4, -0.1, 0.0, 0.2, 0.9, -0.2, 0.3, 0.2, -0.3,
    //     0.2, 0.4, 0.1, -0.3, 0.3, -0.1, -0.3, 1.2, 0.8, 1.2, 0.4, -0.4, -0.1, 1.0, 0.0, -0.4, -0.2,
    //     0.2, 1.0, -1.0, 0.5, 0.3, -0.2, 1.0, 0.3, -0.4, -0.5, 0.6, -1.2, -1.4, -0.7, -1.2, -1.3,
    //     -1.4, -1.3, -1.1, -1.4, -0.9, -1.0, -0.4, -0.0, -0.5, -0.1, -0.4, 0.6, 0.1, 0.9, 0.1, 1.0,
    //     1.5, 1.8, 1.4, 1.7, 1.2, 1.5, 1.2, 1.1, 1.6, 1.2, 0.7, 0.9, -0.3, 0.5, 0.8, -0.2, -0.5,
    //     -0.1, -1.8, -0.4, 0.2, 1.2, 0.3, 0.5, -0.2, -0.7, 0.3, -0.5, -0.2, 0.6, 0.3, -0.7, -0.6,
    //     -1.0, -0.1, 0.0, -0.4, -1.5, -1.0, -0.7, -1.8, -2.9, -0.9, -2.0, -1.0, 0.3, -0.6, -0.6,
    //     0.4, 0.1, -1.4, -0.7, -0.6, -0.9, -0.7, -0.5, -1.2, -0.3, -0.8, -0.8, -0.9, -0.7, -1.1,
    //     -0.1, 0.2, -0.1, 0.3, -0.7, -1.3, -0.7, -0.4, -1.3, -1.2, -1.6, -1.1, -0.6, -1.2, -0.4,
    //     -0.1, 0.8, -0.7, -0.4, 0.1, 0.4, 0.3, 0.3, 0.0, 0.0, -0.5, 1.0, 0.3, 1.1, 0.8, 0.3, 1.2,
    //     0.7, 0.7, 0.6, 0.6, -0.1, 0.9, 0.5, 1.7, -0.4, -0.6, -1.3, -1.4, -0.8, -1.4, -1.4, -1.5,
    //     -1.2, -1.0, -2.7, -2.0, -2.4, -1.4, 0.3, 1.0, 1.2, 1.2, 1.0, 1.1, 1.0, 1.4, 1.8, 1.0, 1.3,
    //     1.4, 0.2, 0.3, 0.5, 0.4, -0.1, 1.0, 1.0, 1.4, 0.7, 1.7, 1.3, 1.2, 0.4, -0.2, -0.2, 0.7,
    //     0.9, 1.1, 1.8, 0.8, 1.0, 1.7, 0.9, 0.2, -0.5, 0.3, -0.2, -0.4, 0.2, -0.0, 0.7, -0.8, 0.4,
    //     1.1, -0.2, -0.1, -0.8, -0.2, -0.5, -1.0, -0.6, -0.4, -0.5, -1.1, -0.2, -0.7, -0.3, -0.1,
    //     -0.3, -0.6, 0.3, 0.1, -0.1, 0.0, -0.3, 1.1, -1.3, 1.2, 0.4, -0.9, 1.0, -0.8, -0.5, -0.3,
    //     -0.3, -0.1, -0.7, -0.8, 0.3, -3.1, 0.3, -0.6, -0.8, 0.4, 0.2, -0.3, 0.4, 1.2, -0.2, -0.0,
    //     1.7, 0.1, 1.8, 1.1, -0.5, -0.2, -0.6, -1.0, -0.6, -1.3, 0.1, -0.3, -0.8, -0.1, 0.2, -0.1,
    //     -0.1, 0.5, -0.3, 0.4, 0.2, 0.7, 0.9, 1.7, 1.8, 2.6, 1.4, 0.7, -0.1, 0.6, 0.3, 1.0, 1.2,
    //     1.3, 1.3, 1.4, 1.1, 1.9, 0.4, 0.8, -0.1, 0.1, 0.2, -0.2, 0.3, -1.2, -0.6, -0.7, -1.1, -1.5,
    //     -0.7, 1.2, 0.9, 0.4, 1.8, 1.8, 2.2, 1.7, 1.3, 2.9, 2.3, 2.7, 2.5, 1.9, 0.4, 0.2, 1.0, 0.4,
    //     1.0, 0.8, 1.1, 2.5, 1.1, 0.5, 0.7, -0.3, 0.0, -0.4, -0.0, -0.2, 0.2, 0.3, 0.3, -0.6, -0.1,
    //     -0.2, 1.5, 0.2, 0.8, 1.2, 0.8, 0.2, 0.3, -0.1, 0.7, 0.1, 1.4, 0.1, -0.9, 0.8, 0.5, 0.2,
    //     -0.2, -0.7, -0.7, -0.6, -0.9, -0.6, -0.8, 0.2, -0.7, -0.0, -0.7, -0.6, -1.1, -1.4, -1.6,
    //     -1.7, -0.5, -0.6, -2.2, -2.0, -0.1, -1.2, 0.4, 0.6, 0.4, 0.7, 1.2, -0.3, -0.1, 0.3, 0.2,
    //     -0.1, 0.9, -0.2, 0.3, -0.4, 0.8, 0.5, 0.6, 0.9, 0.9, -0.1, 1.1, -0.5, 1.5, 0.5, 0.4, -0.1,
    //     0.2, -0.3, -0.9, 0.4, -0.1, 1.0, -0.0, -1.4, -0.3, 0.1, -0.4, -0.5, -0.4, -0.1, -1.2, -0.4,
    //     -0.8, -0.6, 0.2, -0.1, -0.1, 0.2, 0.4,
    // ];
    // println!("{:?}", pacf(&ts, 20))
    // ts = vec![39.,44.,40.,45.,38.,43.,39.];

    // let svm = SSVM {
    //     file_path: "./data_banknote_authentication.txt".to_string(),
    //     drop_column_number: vec![], // just the ID needs to be removed, starts with 1
    //     test_size: 0.20,
    //     learning_rate: 0.000001,
    //     iter_count: 5000,
    //     reg_strength: 50000.,
    // };
    // // println!("Weights are\n{:?}", svm.fit());

    // let weights = svm.fit();
    // println!("Weights : {:?}", weights);
    // let df1 = DataFrame {
    //     string: vec![
    //         vec![
    //             "One", "Two", "Three", "One", "Two", "Three", "One", "Two", "Three", "One", "Two",
    //             "Three",
    //         ],
    //         vec!["1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3"],
    //     ],
    //     numerical: vec![
    //         vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 12., 11.],
    //         vec![
    //             -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12.,
    //         ],
    //     ],
    //     boolean: vec![vec![
    //         true, false, true, true, true, false, true, true, true, false, true, true,
    //     ]],
    // };

    // let df2 = DataFrame {
    //     string: vec![
    //         vec![
    //             "One", "Two", "3", "One", "Two", "3", "One", "Two", "3", "One", "Two", "3",
    //         ],
    //         vec![
    //             "1", "2", "Three", "1", "2", "Three", "1", "2", "Three", "1", "2", "Three",
    //         ],
    //     ],
    //     numerical: vec![
    //         vec![
    //             1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 12.1, 11.1,
    //         ],
    //         vec![
    //             -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12.,
    //         ],
    //     ],
    //     boolean: vec![
    //         vec![
    //             true, false, true, true, true, false, true, true, true, false, true, true,
    //         ],
    //         vec![
    //             true, false, true, true, true, false, true, true, true, false, true, true,
    //         ],
    //     ],
    // };
    // dataframe_comparision(&df1, &df2);
    // use std::collections::HashMap;
    // // creating hashmaps
    // let mut string_columns: HashMap<&str, Vec<&str>> = HashMap::new();
    // string_columns.insert(
    //     "string_1",
    //     vec![
    //         "One", "Two", "Three", "One", "Two", "Three", "One", "Two", "Three", "One", "Two",
    //         "Three",
    //     ],
    // );
    // string_columns.insert(
    //     "string_2",
    //     vec!["1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3"],
    // );
    // let mut numerical_columns: HashMap<&str, Vec<f64>> = HashMap::new();
    // numerical_columns.insert(
    //     "numerical_1",
    //     vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 12., 11.],
    // );
    // numerical_columns.insert(
    //     "numerical_2",
    //     vec![
    //         -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12.,
    //     ],
    // );
    // let mut boolean_columns: HashMap<&str, Vec<bool>> = HashMap::new();
    // boolean_columns.insert(
    //     "boolean_1",
    //     vec![
    //         true, false, true, true, true, false, true, true, true, false, true, true,
    //     ],
    // );

    // let dm1 = DataMap {
    //     string: string_columns,
    //     numerical: numerical_columns,
    //     boolean: boolean_columns,
    // };

    // // second datamap
    // let mut string_columns: HashMap<&str, Vec<&str>> = HashMap::new();
    // string_columns.insert(
    //     "string_1",
    //     vec![
    //         "One", "Two", "3", "One", "Two", "3", "One", "Two", "3", "One", "Two", "3",
    //     ],
    // );
    // string_columns.insert(
    //     "string_2",
    //     vec![
    //         "One", "2", "3", "One", "2", "3", "One", "2", "3", "One", "2", "3",
    //     ],
    // );
    // let mut numerical_columns: HashMap<&str, Vec<f64>> = HashMap::new();
    // numerical_columns.insert(
    //     "numerical_1",
    //     vec![
    //         1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 12.1, 11.1,
    //     ],
    // );
    // numerical_columns.insert(
    //     "numerical_2",
    //     vec![
    //         -1.1, -2., -3.1, -4., -5.1, -6., -7.1, -8., -9.1, -10., -11.1, -12.,
    //     ],
    // );
    // numerical_columns.insert(
    //     "numerical_5",
    //     vec![
    //         -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12.,
    //     ],
    // );
    // let mut boolean_columns: HashMap<&str, Vec<bool>> = HashMap::new();
    // boolean_columns.insert(
    //     "boolean_1",
    //     vec![
    //         true, false, true, true, true, false, true, true, true, false, true, true,
    //     ],
    // );

    // let dm2 = DataMap {
    //     string: string_columns,
    //     numerical: numerical_columns,
    //     boolean: boolean_columns,
    // };
    // datamap_comparision(&dm1, &dm2);
    // let data = vec![
    //     vec![10., 11., 8., 3., 2., 1.],
    //     vec![6., 4., 5., 3., 2.8, 1.],
    //     vec![12., 9., 10., 2.5, 1.3, 2.],
    //     vec![5., 7., 6., 2., 4., 7.],
    // ];
    // let mut data = vec![
    //     vec![13., -20., 3.],
    //     vec![1., -7.5, 13.3],
    //     vec![3.5, -10., 1.],
    // ];
    // let (_, values) = read_csv("dataset_iris.txt".to_string());
    // let mut data = values
    //     .iter()
    //     .map(|a| a[..a.len() - 1].to_vec())
    //     .collect::<Vec<Vec<String>>>(); // removing the lables
    // let iris = float_randomize(&data);
    // // print_a_matrix("Iris", &iris);
    // pca(&iris, 1);
    // data = vec![
    //     vec![10., 11., 8., 3., 2., 1.],
    //     vec![6., 4., 5., 3., 2.8, 1.],
    // ];
    // println!("{:?}", best_fit_line_in_higher_dimension(&data));
    // let mut df2 = DataFrame {
    //     string: vec![
    //         vec![
    //             "One", "Two", "3", "One", "Two", "3", "One", "Two", "3", "One", "Two", "3",
    //         ],
    //         vec![
    //             "1", "2", "Three", "1", "2", "Three", "1", "2", "Three", "1", "2", "Three",
    //         ],
    //     ],
    //     numerical: vec![
    //         vec![
    //             1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 12.1, 11.1,
    //         ],
    //         vec![
    //             -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12.,
    //         ],
    //     ],
    //     boolean: vec![
    //         vec![
    //             true, false, true, true, true, false, true, true, true, false, true, true,
    //         ],
    //         vec![
    //             true, false, true, true, true, false, true, true, true, false, true, true,
    //         ],
    //     ],
    // };
    // let df2 = df2.sort("n", 1, true);
    // println!("{:?}", df2.string);
    // println!("{:?}", df2.numerical);
    // println!("{:?}", df2.boolean);

    let mut string_columns: HashMap<&str, Vec<&str>> = HashMap::new();
    string_columns.insert(
        "string_1",
        vec![
            "One", "Two", "Three", "One", "Two", "Three", "One", "Two", "Three", "One", "Two",
            "Three",
        ],
    );
    string_columns.insert(
        "string_2",
        vec!["1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3"],
    );
    let mut numerical_columns: HashMap<&str, Vec<f64>> = HashMap::new();
    numerical_columns.insert(
        "numerical_1",
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 12., 11.],
    );
    numerical_columns.insert(
        "numerical_2",
        vec![
            -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12.,
        ],
    );
    let mut boolean_columns: HashMap<&str, Vec<bool>> = HashMap::new();
    boolean_columns.insert(
        "boolean_1",
        vec![
            true, false, true, true, true, false, true, true, true, false, true, true,
        ],
    );

    let dm = DataMap {
        string: string_columns,
        numerical: numerical_columns,
        boolean: boolean_columns,
    };
    println!("SORTING (Ascending):");
    let dm1 = dm.sort("s", "string_1", true);
    println!("{:?}", dm1.string);
    println!("{:?}", dm1.numerical);
    println!("{:?}", dm1.boolean);
    println!("SORTING (Descending):");
    let dm1 = dm.sort("s", "string_1", false);
    println!("{:?}", dm1.string);
    println!("{:?}", dm1.numerical);
    println!("{:?}", dm1.boolean);
}
use std::collections::HashMap;
pub struct DataMap<'a> {
    // use std::collections::HashMap;
    // stored column wise
    pub string: HashMap<&'a str, Vec<&'a str>>,
    pub numerical: HashMap<&'a str, Vec<f64>>,
    pub boolean: HashMap<&'a str, Vec<bool>>,
}
impl<'a> DataMap<'a> {
    pub fn sort(&self, col_type: &str, col_name: &str, ascending: bool) -> DataMap {
        /* returns a different DataFrame with rows sorted as per order passed
        col_type : "s": string,"n":numerical
        */
        let mut output = DataMap {
            string: HashMap::new(),
            numerical: HashMap::new(),
            boolean: HashMap::new(),
        };
        let mut to_sort_by_string;
        let mut to_sort_by_numerical;
        let order: Vec<usize>;
        match col_type {
            "s" => {
                to_sort_by_string = self.string[col_name].clone();
                // finding the order of sorting
                order = DataFrame::find_order_of_sorting_string(&mut to_sort_by_string, ascending);
            }
            "n" => {
                to_sort_by_numerical = self.numerical[col_name].clone();
                // finding the order of sorting
                order = DataFrame::find_order_of_sorting_numerical(
                    &mut to_sort_by_numerical,
                    ascending,
                );
            }
            _ => panic!("Pass either `s` or `n`"),
        }

        println!("New order is : {:?}", order);
        // reordering the original DataFrame (String)
        for (key, value) in self.string.iter() {
            let mut new_vector = vec![];
            for o in order.iter() {
                new_vector.push(value[*o]);
            }
            output.string.insert(*key, new_vector);
        }
        // reordering the original DataFrame (Numerical)
        for (key, value) in self.numerical.iter() {
            let mut new_vector = vec![];
            for o in order.iter() {
                new_vector.push(value[*o]);
            }
            output.numerical.insert(*key, new_vector);
        }
        // reordering the original DataFrame (Numerical)
        for (key, value) in self.boolean.iter() {
            let mut new_vector = vec![];
            for o in order.iter() {
                new_vector.push(value[*o]);
            }
            output.boolean.insert(*key, new_vector);
        }

        output
    }

    fn find_order_of_sorting_string(data: &mut Vec<&str>, ascending: bool) -> Vec<usize> {
        use std::collections::BTreeMap;
        let mut input = data.clone();
        let mut order: BTreeMap<usize, &str> = BTreeMap::new();
        let mut output = vec![];

        // original order
        for (n, i) in data.iter().enumerate() {
            order.insert(n, i);
        }
        // println!("{:?}", order);
        match ascending {
            true => input.sort_unstable(),
            false => {
                input.sort_unstable();
                input.reverse();
            }
        };

        // new order
        for i in input.iter() {
            for (k, v) in order.iter() {
                if (*i == *v) & (output.contains(k) == false) {
                    output.push(*k);
                    break;
                }
            }
        }
        output
    }

    fn find_order_of_sorting_numerical(data: &mut Vec<f64>, ascending: bool) -> Vec<usize> {
        use std::collections::BTreeMap;
        let mut input = data.clone();
        let mut order: BTreeMap<usize, &f64> = BTreeMap::new();
        let mut output = vec![];

        // original order
        for (n, i) in data.iter().enumerate() {
            order.insert(n, i);
        }
        // println!("{:?}", order);
        match ascending {
            true => input.sort_by(|a, b| a.partial_cmp(b).unwrap()),
            false => input.sort_by(|a, b| b.partial_cmp(a).unwrap()),
        };

        // new order
        for i in input.iter() {
            for (k, v) in order.iter() {
                if (i == *v) & (output.contains(k) == false) {
                    output.push(*k);
                    break;
                }
            }
        }
        output
    }
}

// ================================================================================= // =================================================================================
// ================================================================================= // =================================================================================
// ================================================================================= // =================================================================================
// ================================================================================= // =================================================================================// ================================================================================= // =================================================================================
// ================================================================================= // =================================================================================
// https://towardsdatascience.com/detecting-stationarity-in-time-series-data-d29e0a21e638 https://github.com/2wavetech/How-to-Check-if-Time-Series-Data-is-Stationary-with-Python
// https://towardsdatascience.com/principal-component-analysis-pca-from-scratch-in-python-7f3e2a540c51
// pub fn pca(data: &Vec<Vec<f64>>, reduce_to: usize) -> Vec<Vec<f64>> {
//     /*
//      Steps
//      1. Scale the data
//      2. Find covariance matrix
//      3. Eigendecomposition
//     */
//     /*
//     Video: https://www.youtube.com/watch?v=FgakZw6K1QQ, https://www.youtube.com/watch?v=PFDu9oVAE-g
//     Article: https://towardsdatascience.com/principal-component-analysis-pca-from-scratch-in-python-7f3e2a540c51, https://medium.com/@louisdevitry/intuitive-tutorial-on-eigenvalue-decomposition-in-numpy-af0062a4929b
//     */
//     println!("Original data");
//     head(&data, 5);
//     let mut output = vec![];
//     let mut input = data.clone();
//     let mut cov_mat = vec![];

//     // 1. standardize data and transpose it
//     input = transpose(&columns_to_rows_conversion(
//         &row_to_columns_conversion(&input)
//             .iter()
//             .map(|a| standardize_vector_f(a))
//             .collect(),
//     ));

//     println!("Data Standardized .. ");

//     // 2. Covariance matrix (ASSUMING DATA IS ROW WISE)
//     for (n, i) in input.iter().enumerate() {
//         let mut row = vec![];
//         for (m, j) in input.iter().enumerate() {
//             if n == m {
//                 row.push(round_off_f(variance(i) / input[0].len() as f64, 3));
//             } else {
//                 row.push(round_off_f(covariance(i, j) / input[0].len() as f64, 3));
//             }
//         }
//         cov_mat.push(row);
//     }
//     print_a_matrix("Covariance Matrix", &cov_mat);

//     // 3. Eigen decomposition
//     eigen_decomposition(&cov_mat);
//     output
// }
// pub fn eigen_decomposition(matrix: &Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
//     //
//     // returns eigen values and eigen vectors
//     let mut eigen_values = vec![];
//     let mut eigen_vectors = vec![];

//     (eigen_values, eigen_vectors)
// }

// pub fn make_string_float(matrix: &Vec<Vec<String>>) -> Vec<Vec<f64>> {
//     // similar to float_ramdomize without randomizing
//     matrix
//         .iter()
//         .map(|a| {
//             a.iter()
//                 .map(|b| (*b).replace("\r", "").parse::<f64>().unwrap())
//                 .collect::<Vec<f64>>()
//         })
//         .collect::<Vec<Vec<f64>>>()
// }

// pub fn pca(data: &Vec<Vec<f64>>, reduce_to: usize) -> Vec<Vec<f64>> {
//     /*
//     Steps:
//     1. Average of each row (assuming data is row wise)
//     2. Find the point which is the center and move it to origin
//     3. Find the best fit line's equation
//     4. Find the most imp column on PC1
//     */
//     /*
//     Video: https://www.youtube.com/watch?v=FgakZw6K1QQ, https://www.youtube.com/watch?v=PFDu9oVAE-g
//     Article: https://towardsdatascience.com/principal-component-analysis-pca-from-scratch-in-python-7f3e2a540c51
//     */
//     print_a_matrix("Original data", &data);
//     let mut output = vec![];
//     let mut center_of_data = vec![];
//     let mut input = data.clone();
//     let mut shifted_values = input.clone();
//     // 1. Average of each row (assuming data is row wise)
//     for each_row in input.iter() {
//         center_of_data.push(mean(each_row));
//     }
//     println!("The center of the data :{:?}", center_of_data);

//     // 2. Find the point which is the center and move it to origin
//     for (row_count, each_row) in input.iter().enumerate() {
//         for (center_count, center_value) in center_of_data.iter().enumerate() {
//             shifted_values[row_count] = each_row.iter().map(|a| a + (-1. * center_value)).collect()
//         }
//     }
//     print_a_matrix("Shifted input is:", &shifted_values);
//     let number_of_rows = shifted_values.len();

//     // 3. Find the best fit line's equation
//     let xt = MatrixF {
//         matrix: shifted_values,
//     };
//     let xtx = MatrixF {
//         matrix: matrix_multiplication(&xt.matrix, &transpose(&xt.matrix)),
//     };
//     // println!("{:?}", MatrixF::inverse_f(&xtx));
//     let slopes = &matrix_multiplication(
//         &MatrixF::inverse_f(&xtx), // np.linalg.inv(X.T@X)
//         // &transpose(&vec![matrix_vector_product_f(&xt.matrix,&vec![1.; number_of_rows])]), //(X.T)
//         &transpose(&xt.matrix),
//     )[0];
//     println!("The importance of each column : {:?}", slopes);

//     output
// }

// pub fn line_between_2_points(point1: &Vec<f64>, point2: &Vec<f64>) {
//     let direction_vector = element_wise_operation(point1, point2, "sub");
//     println!("Directional vector : {:?}", direction_vector);
// }

// pub fn best_fit_line_in_higher_dimension(data: &Vec<Vec<f64>>) -> Vec<f64> {
//     // returns coefficients in the form of ax+by+cz+...+n
//     // assuming data is passed row wise
//     // https://www.youtube.com/watch?v=U4eRSL16KzA
//     print_a_matrix("The original matrix is:", &data);
//     let mut input = row_to_columns_conversion(&data);
//     let column_count = input.len() - 1;
//     input[column_count] = vec![1.; input[0].len()];
//     let a = MatrixF {
//         matrix: input.clone(),
//     };
//     print_a_matrix("The modified input matrix is:", &a.matrix);
//     print_a_matrix("Inverse", &a.inverse_f());
//     matrix_vector_product_f(&a.inverse_f(), &input[column_count].to_vec())
//         .iter()
//         .map(|a| round_off_f(*a, 2))
//         .collect()
// }

pub fn pacf(ts: &Vec<f64>, lag: usize) -> Vec<f64> {
    /*
    Unlike ACF, which uses combined effect on a value, here the impact is very specific
    The coeeficient is specific to the lagged value
    This is more useful than ACF as it remoevs the influence of values in between
    */
    // data: https://www.ncdc.noaa.gov/teleconnections/enso/indicators/soi/data.csv
    // https://www.youtube.com/watch?v=ZjaBn93YPWo, http://rinterested.github.io/statistics/acf_pacf.html, https://towardsdatascience.com/understanding-partial-auto-correlation-fa39271146ac
    /*
    STEPS:
    1. While shifting the ts from front, and residual from last (residual is same as ts initially)
    2. From the best fit line between ts and residual find the new residual
    3. From now on best fit line will be found on the residual
    4. The correlation between shifted ts and point residual at each shift will be captured as pacf at that point
    */
    let mut pacf = vec![1.]; // as correlation with it self is 1
    let residual = ts.clone();
    for i in 1..lag {
        let mut res_shift = &residual[i..].to_vec();
        // finding correlation
        let corr = correlation(&ts[..(ts.len() - i)].to_vec(), &res_shift, "p");
        pacf.push(corr);
        // calculting best fit line
        let (intercept, slope) =
            best_fit_line(&ts[..(ts.len() - i)].to_vec(), &residual[i..].to_vec());
        // calculating estimate
        let estimate = &ts[..(ts.len() - i)]
            .to_vec()
            .iter()
            .map(|a| (a * slope) + intercept)
            .collect::<Vec<f64>>();
        // modifying residual to act as source data in the next lag
        res_shift = &res_shift
            .iter()
            .zip(estimate.iter())
            .map(|a| a.0 - a.1)
            .collect::<Vec<f64>>();
        println!("slope : {:?}, intercept : {:?}", slope, intercept);
    }

    pacf
}

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

// pub fn lagmat<T: Copy>(array: &Vec<T>, lag: u32) -> Vec<Vec<T>> {
//     // https://gist.github.com/jcorrius/c3212b991b4f484cd502a50e7b92d41b
//     let mut output = vec![];
//     for (n, _) in array.iter().enumerate() {
//         if n < array.len() - (lag as usize) {
//             let mut subarray = array[n..(n + 1 + (lag as usize))].to_vec();
//             subarray.reverse();
//             output.push(subarray);
//         }
//     }
//     output
// }

// pub fn lag_n_diff_f(array: &mut Vec<f64>, lag: usize) -> Vec<f64> {
//     // https://gist.github.com/jcorrius/c3212b991b4f484cd502a50e7b92d41b
//     let lag_temp: Vec<f64> = pad_with_zero(&mut array[lag..].to_vec(), lag, "post");
//     let mut diff_temp = element_wise_operation(&lag_temp, &array, "sub");
//     diff_temp = diff_temp.iter().map(|a| round_off_f(*a, 1)).collect();
//     diff_temp = diff_temp[..diff_temp.len() - lag].to_vec();
//     diff_temp
// }

// fn find_best_split<T>(X: Vec<Vec<T>>, Y: Vec<T>) {}

// Classification tree
// https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775
// https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
// https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea

// pub struct TimeSeries<T> {
//     data: Vec<T>,
// }
// impl<T> TimeSeries<T>
// where
//     T: std::clone::Clone,
// {
//     pub fn lag_view(self, window_size: usize) -> (Vec<Vec<T>>, Vec<T>) {
//         /*
//             Returns sliding window of data and lagged data
//         */
//         // https://www.ritchievink.com/blog/2018/09/26/algorithm-breakdown-ar-ma-and-arima-models/
//         let mut window_matrix = vec![];
//         let reduced_data = self.data[window_size..].to_vec();
//         for (n, _) in self.data[..self.data.len() - window_size]
//             .iter()
//             .enumerate()
//         {
//             window_matrix.push(self.data[n..n + window_size].to_vec());
//         }
//         (window_matrix, reduced_data)
//     }

//     pub fn ma(epsilon: Vec<T>, theta: Vec<T>) {}
// }
