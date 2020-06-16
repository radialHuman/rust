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
    //     BinaryLogisticRegression_f::sigmoid_activation(&vec![
    //         0.19849274, 0.21902214, 0.68417234, 0.13458896, 0.61311337, 0.08175402, 0.52679541,
    //         0.5219286, -1.4237689, 0.98864689, 0.79766791, 0.23862188
    //     ])
    // );

    // let ts = TimeSeries {
    //     data: vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
    // };
    // println!("{:?}", ts.lag_view(3));

    // reading in data and splitting it into features and target
    // let (x, y) = BinaryLogisticRegression_f::read_n_split_n_shuffle(
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
    // let blr = BinaryLogisticRegression_f {
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
}

// LOGISTIC REGRESSION



// ================================================================================= // =================================================================================
// ================================================================================= // =================================================================================
// ================================================================================= // =================================================================================
// ================================================================================= // =================================================================================// ================================================================================= // =================================================================================
// ================================================================================= // =================================================================================

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
