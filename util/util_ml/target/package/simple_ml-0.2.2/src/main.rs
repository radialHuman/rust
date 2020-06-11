/*
To have a lib_matrixrary and neural_network_library of all the functions related to nn in one palce
Derived from nnfs
May turn into a crate with generics
*/
mod lib_matrix;
mod lib_nn;
use lib_nn::LayerDetails;
mod lib_ml;

use rand::*;

fn main() {
    // lib_matrix.rs

    let v1 = vec![1., 2., 5., 8., 3., 7., 5., -2., -3., -5.];
    let mut v2 = vec![1, 6, 2, 7, 3, 7, 99, 6, 3, 6];

    // vector addition check
    let v1_plus_v2 = lib_matrix::element_wise_operation(
        &mut v1.iter().map(|x| *x as i32).collect(),
        &mut v2,
        "Add",
    );
    println!(
        "The addition of {:?} with {:?} is: {:?}",
        v1, v2, v1_plus_v2
    );

    // elemnt wise multiplicaiton check
    let v1_into_v2 = lib_matrix::element_wise_operation(
        &mut v1.iter().map(|x| *x as i32).collect(),
        &mut v2,
        "Mul",
    );
    println!(
        "The multiplication of {:?} with {:?} is: {:?}",
        v1, v2, v1_into_v2
    );

    v2 = vec![1, 6, 2, 7, 3, 7, 99, 6, 3, 6, -12, -34];
    // Shape changer
    let v2 = lib_matrix::shape_changer(&v2, 4, 3);
    lib_matrix::print_a_matrix("\n2x5 version is:", &v2);

    // matrix transpose check
    let v2_t = lib_matrix::transpose(&v2);
    lib_matrix::print_a_matrix("The previous matrix is transposed to", &v2_t);

    // matrix multiplcation and dot product
    let v3 = vec![vec![1, 4, 4], vec![5, 8, 9], vec![0, 1, 6]];
    let v4 = vec![vec![1, 4, 4, 5], vec![5, 8, 9, 1], vec![0, 1, 6, 0]];
    let v3_v4 = lib_matrix::matrix_multiplication(&v3, &v4);
    lib_matrix::print_a_matrix(
        &format!("The multiplicaiton of {:?} and {:?}", v3, v4),
        &v3_v4,
    );

    //================================================================================================================
    section_break("MATRIX OVER"); // lib_nn.rs
                                  // takes in only f64

    // ACTIVATION FUCNTION ReLU
    println!("ReLU of {:?} is {:?}", v1, lib_nn::activation_relu(&v1));

    // ACTIVATION FUNCTION Leaky ReLU
    println!(
        "Leaky ReLU of {:?} with alpha 0.1 is {:?}",
        v1,
        lib_nn::activation_leaky_relu(&v1, 0.1)
    );

    // ACTIVATION FUNCTION Sigmoid
    println!(
        "Sigmoid of {:?} is {:?}",
        v1,
        lib_nn::activation_sigmoid(&v1)
    );

    // ACTIVATION FUNCTION TanH
    println!("TanH of {:?} is {:?}", v1, lib_nn::activation_tanh(&v1));

    // Creating neuron
    let n_features = 6;
    let layer_1 = LayerDetails {
        n_inputs: n_features,
        n_neurons: 5, // can be of any size, depends
    };

    // Creating input
    let mut input = vec![];
    let data_points = 10;
    let mut rng = thread_rng();
    for _ in 0..data_points {
        let mut row = vec![];
        for _ in 0..layer_1.n_inputs {
            row.push(rng.gen_range(-10., 10.));
        }
        input.push(row);
    }
    lib_matrix::print_a_matrix("\nInput generated is :", &input);

    // Output of a layer activation_function(input*weights+bias)
    let output = layer_1.output_of_layer(
        &input,
        &layer_1.create_weights(),
        &mut layer_1.create_bias(),
        lib_nn::activation_relu, // to be choosen by user, when there are other fucntiosn
    );

    lib_matrix::print_a_matrix("\nOutput generated is :", &output);

    //================================================================================================================
    section_break("NN OVER");
    // lib_ml.rs
    let v1 = vec![1., 2., 4., 3., 5.];
    let v2 = vec![1., 3., 3., 2., 5.];
    println!("Mean of {:?} is {}", &v1, lib_ml::mean(&v1));
    println!("variance of {:?} is {}", &v1, lib_ml::variance(&v1));
    println!("Mean of {:?} is {}", &v1, lib_ml::mean(&v2));
    println!("variance of {:?} is {}", &v1, lib_ml::variance(&v2));
    println!(
        "The covariance of {:?} and {:?} is {}",
        &v1,
        &v2,
        lib_ml::covariance(&v1, &v2)
    );
    println!(
        "Coefficient of {:?} and {:?} are b0 = {} and b1 = {}",
        &v1,
        &v2,
        lib_ml::coefficient(&v1, &v2).0,
        lib_ml::coefficient(&v1, &v2).1
    );

    // Simple linear regression
    let to_train_on = vec![
        (1., 2.),
        (2., 3.),
        (4., 5.),
        (3., 5.),
        (6., 8.),
        (7., 8.),
        (9., 10.),
        (1., 2.5),
        (11., 12.),
        (5., 4.),
        (7., 7.),
        (6., 6.),
        (8., 9.),
    ];
    let to_test_on = vec![(10., 11.), (9., 12.), (11., 12.5)];
    let predicted_output = lib_ml::simple_linear_regression_prediction(&to_train_on, &to_test_on);
    let original_output: Vec<_> = to_test_on.iter().map(|a| a.0).collect();
    println!(
        "Predicted is {:?}\nOriginal is {:?}",
        &predicted_output, &original_output
    );

    // reading in a file to have table
    let df = lib_ml::read_csv("./data/dataset_iris.txt".to_string());
    println!("{:?}\n{:?}", df.0, df.1);

    // unique values
    println!(
        "Unique classes are {:?}",
        lib_matrix::unique_values(&vec![1, 6, 2, 6, 8, 2, 23, 3, 5, 2, 4, 2, 0]) // converting String to &str as copy is not implemented for String
    );

    // // type conversion and missing value replacement
    // let conversion =
    //     lib_ml::convert_and_impute(&vec![1, 6, 2, 6, 8, 2, 23, 3, 5, 2, 4, 2, 0], 0., 999.);
    // let floating_petal_length = conversion.0.unwrap();
    // let missing_value = conversion.1;
    // println!(
    //     "{:?}\nis now\n{:?}\nwith missing values at\n{:?}",
    //     &vec![1, 6, 2, 6, 8, 2, 23, 3, 5, 2, 4, 2, 0],
    //     floating_petal_length,
    //     missing_value
    // );

    // // missing string imputation
    // let mut species = vec![1, 6, 2, 6, 8, 2, 23, 3, 5, 2, 4, 2, 0].clone();
    // println!(
    //     "{:?}\nis now\n{:?}",
    //     &&vec![1, 6, 2, 6, 8, 2, 23, 3, 5, 2, 4, 2, 0],
    //     lib_ml::impute_string(&mut species, "UNKNOWN")
    // );

    // // unique values
    // println!(
    //     "Now the unique classes are {:?}",
    //     lib_matrix::unique_values(&lib_ml::impute_string(&mut species, "UNKNOWN")) // converting String to &str as copy is not implemented for String
    // );

    // // string to categories
    // println!(
    //     "Categorized {:?} is now {:?}",
    //     &df["species"],
    //     lib_ml::convert_string_categorical(&df["species"].iter().map(|a| &*a).collect(), false)
    // );

    // // unique value count (pivot)
    // println!(
    //     "The value count of SPECIES is {:?}",
    //     lib_matrix::value_counts(&df["species"].iter().map(|a| &*a).collect())
    // );

    // maximum and minimum
    let arr = vec![1., 2., 3., -5., -7., 0.];
    let (min, max) = lib_matrix::min_max_f(&arr);
    println!("In {:?}\nminimum is {} and maximum is {}", arr, min, max);

    // normalize vector
    println!(
        "{:?}\nNormalized to : {:?}",
        arr,
        lib_ml::normalize_vector_f(&arr)
    );

    // logistic and make matrix float
    let matrix = vec![vec![1, 2], vec![2, 3], vec![3, 7], vec![34, 76]];
    let beta = vec![vec![0.2, 0.3], vec![0.4, 0.7], vec![1., 2.], vec![0.6, 0.]];
    println!(
        "{:?}\n{:?}\nBecomes {:?}\n using logistic fucntion ",
        matrix,
        beta,
        lib_ml::logistic_function_f(&lib_matrix::make_matrix_float(&matrix), &beta)
    );

    // make vector float
    println!(
        "{:?}\n converted to float: {:?}",
        matrix[0],
        lib_matrix::make_vector_float(&matrix[0])
    );

    // round off
    println!(
        "Rounding of {} by {} = {}",
        3.14267864,
        4,
        lib_matrix::round_off_f(3.14267864, 4)
    );

    // cost function
    let a_f = vec![vec![1., 2.], vec![2., 3.], vec![3., 7.], vec![4., 6.]];
    let mut b = vec![vec![0.2, 0.3], vec![0.4, 0.7], vec![1., 2.], vec![0.6, 1.]];
    let y = vec![5., 6., 1., 2.];
    // println!("{:?}", lib_ml::cost_function_f(&a_f, &b, &y));

    // matrix addition
    println!(
        "{:?}",
        lib_matrix::element_wise_matrix_operation(&a_f, &b, "Add")
    );

    // matrix subtraction
    println!(
        "{:?}",
        lib_matrix::element_wise_matrix_operation(&a_f, &b, "Sub")
    );

    // element wise matrix multiplication
    println!(
        "{:?}",
        lib_matrix::element_wise_matrix_operation(&a_f, &b, "Mul")
    );

    // gradient descent
    // println!(
    //     "{:?}",
    //     lib_ml::gradient_descent(&a_f, &mut b, &y, 0.01, 0.001)
    // );

    let (columns, values) = lib_ml::read_csv("ccpp.csv".to_string());
    let mlr = lib_ml::MultivariantLinearRegression {
        header: columns,
        data: values,
        split_ratio: 0.25,
        alpha_learning_rate: 0.005,
        iterations: 1000,
    };
    mlr.multivariant_linear_regression();

    //================================================================================================================
    section_break("ML OVER");
}

pub fn section_break(display: &str) {
    println!("");
    println!("");
    println!("================================================================================================================================================================");
    println!("\t\t\t\t\t\t\t\t\t\t{:?}", display);
    println!("================================================================================================================================================================");
    println!("");
    println!("");
}

/* OUTPUT
The addition of [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, -2.0, -3.0, -5.0] with [1, 6, 2, 7, 3, 7, 99, 6, 3, 6] is: [2, 8, 7, 15, 6, 14, 104, 4, 0, 1]
The multiplication of [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, -2.0, -3.0, -5.0] with [1, 6, 2, 7, 3, 7, 99, 6, 3, 6] is: [1, 12, 10, 56, 9, 49, 495, -12, -9, -30]

2x5 version is:
[1, 6, 2, 7]
[3, 7, 99, 6]
[3, 6, -12, -34]


The previous matrix is transposed to
[1, 3, 3]
[6, 7, 6]
[2, 99, -12]
[7, 6, -34]


The multiplicaiton of [[1, 4, 4], [5, 8, 9], [0, 1, 6]] and [[1, 4, 4, 5], [5, 8, 9, 1], [0, 1, 6, 0]]
[21, 40, 64]
[9, 45, 93]
[146, 33, 5]
[14, 45, 1]




================================================================================================================================================================
                                                                                "MATRIX OVER"
================================================================================================================================================================


ReLU of [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, -2.0, -3.0, -5.0] is [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, 0.0, 0.0, 0.0]
Leaky ReLU of [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, -2.0, -3.0, -5.0] with alpha 0.1 is [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, -0.2, -0.30000000000000004, -0.5]
Sigmoid of [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, -2.0, -3.0, -5.0] is [0.2689414213699951, 0.11920292202211755, 0.0066928509242848554, 0.0003353501304664781, 0.04742587317756678, 0.0009110511944006454, 0.0066928509242848554, 0.8807970779778823, 0.9525741268224334, 0.9933071490757153]
TanH of [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, -2.0, -3.0, -5.0] is [0.7615941559557649, 0.964027580075817, 0.999909204262595, 0.999999774929676, 0.9950547536867306, 0.9999983369439447, 0.999909204262595, -0.964027580075817, -0.9950547536867306, -0.999909204262595]

Input generated is :
[-5.706055277627109, 4.485154906629173, -4.128299199456578, -3.556326425602414, 8.839404968848207, -2.1909442121404155]
[0.14475201826447126, 0.6518311151851499, -1.4722508981520512, -2.766597699497453, -7.566998247774572, 5.360896445047025]
[8.15248052676504, -8.868135724403835, 1.1542290537956603, -0.046209687983921555, 7.399819110111174, -2.2272988134226024]
[-6.5866411770572775, -7.501109480479786, -8.907638934743382, -4.867501718709932, -0.7262001309995156, -8.567022296658372]
[-4.893344698638886, -6.850978470713454, -4.14248228518328, 3.7855493911210303, -4.800722720226069, -8.588043831815678]
[-2.859695149983983, -8.733753782040306, -3.782236734389528, -5.149187846011047, -9.645534965268773, -3.973771078764603]
[-9.064006104956235, -8.950653054798705, 5.527765570581478, -5.734565621030345, -1.4596845829272898, 8.527905203694868]
[2.4367551618852765, 5.001406118489573, -1.886553341913345, -9.31174340958949, 1.9661145993044968, -9.257753775824229]
[-2.16625529328776, 7.91346840826316, -7.774440795605995, -0.16097834474748218, -6.676376209015902, -2.9809026740751987]
[4.643637601645757, 9.800121077593907, 5.485281233728397, -8.540843684837917, -4.879647490956418, -1.807116473679372]



Output generated is :
[1.5317770486451892, 13.862988415624201, 0.0, 0.0, 0.0]
[1.2465993412250649, 1.735863606855439, 0.0, 0.0, 3.903399137395363]
[0.0, 7.08882065343584, 0.0, 13.907297872489101, 0.0]
[0.555636845036233, 0.0, 5.7945445039211725, 0.0, 18.288724811515248]
[0.7114990653821007, 20.61318847894542, 0.0, 0.0, 0.0]
[0.0, 2.209463020689462, 0.0, 13.929804947945488, 6.149269324054226]
[2.0263842569567925, 0.0, 0.0, 2.79220653533398, 5.782852353229802]
[3.782181581986737, 0.0, 2.733375010947444, 0.0, 4.872665389563409]
[2.82758818431387, 2.383502356542455, 1.0225637229482705, 4.875051614136698, 4.718897094829307]
[0.0, 3.4079452107031485, 0.0, 6.667742940297414, 0.0]




================================================================================================================================================================
                                                                                "NN OVER"
================================================================================================================================================================


Mean of [1.0, 2.0, 4.0, 3.0, 5.0] is 3
variance of [1.0, 2.0, 4.0, 3.0, 5.0] is 10
Mean of [1.0, 2.0, 4.0, 3.0, 5.0] is 2.8
variance of [1.0, 2.0, 4.0, 3.0, 5.0] is 8.8
The covariance of [1.0, 2.0, 4.0, 3.0, 5.0] and [1.0, 3.0, 3.0, 2.0, 5.0] is 8
Coefficient of [1.0, 2.0, 4.0, 3.0, 5.0] and [1.0, 3.0, 3.0, 2.0, 5.0] are b0 = 0.39999999999999947 and b1 = 0.8
========================================================================================================================================================
RMSE: 2.080646271630258
Predicted is [11.92023172905526, 11.901403743315509, 11.93905971479501]
Original is [10.0, 9.0, 11.0]
========================================================================================================================================================
Reading the file ...
Input row count is 30
The header is ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
[["1.4", "1.4", "1.3", "1.5", "1.4", "1.7", "1.4", "1.5", "1.4", "1.5", "1.5", "1.6", "1.4", "1.1", "1.2", "1.5", "1.3", "1.4", "1.7", "1.5", "1.7", "1.5", "1.0", "1.7", "1.9", "1.6", "1.6", "1.5", "1.4", "1.6", "1.6", "1.5", "1.5", "1.4", "1.5", "1.2", "1.3", "1.5", "1.3", "1.5", "1.3", "1.3", "1.3", "1.6", "1.9", "1.4", "1.6", "1.4", "1.5", "1.4", "4.7", "4.5", "4.9", "4.0", "4.6", "4.5", "4.7", "3.3", "4.6", "3.9", "3.5", "4.2", "4.0", "4.7", "3.6", "4.4", "4.5", "4.1", "4.5", "3.9",
"4.8", "4.0", "4.9", "4.7", "4.3", "4.4", "4.8", "5.0", "4.5", "3.5", "3.8", "3.7", "3.9", "5.1", "4.5", "4.5", "4.7", "4.4", "4.1", "4.0", "4.4", "4.6", "4.0", "3.3", "4.2", "4.2", "4.2", "4.3", "3.0", "4.1", "6.0", "5.1", "5.9", "5.6", "5.8", "6.6", "4.5", "6.3", "5.8", "6.1", "5.1", "5.3", "5.5", "5.0", "5.1", "5.3", "5.5", "6.7", "6.9", "5.0", "5.7", "4.9", "6.7", "4.9", "5.7", "6.0", "4.8", "4.9", "5.6", "5.8", "6.1", "6.4", "5.6", "5.1", "5.6", "6.1", "5.6", "5.5", "4.8", "5.4", "5.6", "5.1", "5.1", "5.9", "5.7", "5.2", "5.0", "5.2", "5.4", "5.1"], ["5.1", "4.9", "4.7", "4.6", "5.0", "5.4", "4.6", "5.0", "4.4", "4.9", "5.4", "4.8", "4.8", "4.3", "5.8", "5.7", "5.4", "5.1", "5.7", "5.1", "5.4", "5.1", "4.6", "5.1", "4.8", "5.0", "5.0", "5.2", "5.2", "4.7", "4.8", "5.4", "5.2", "5.5", "4.9", "5.0", "5.5", "4.9", "4.4", "5.1", "5.0", "4.5", "4.4", "5.0", "5.1", "4.8", "5.1", "4.6", "5.3", "5.0", "7.0", "6.4", "6.9", "5.5", "6.5", "5.7", "6.3", "4.9", "6.6", "5.2", "5.0", "5.9", "6.0", "6.1", "5.6", "6.7", "5.6", "5.8", "6.2", "5.6", "5.9", "6.1", "6.3", "6.1", "6.4", "6.6", "6.8", "6.7", "6.0", "5.7", "5.5", "5.5", "5.8", "6.0", "5.4", "6.0", "6.7", "6.3", "5.6", "5.5", "5.5", "6.1", "5.8", "5.0", "5.6", "5.7", "5.7", "6.2", "5.1", "5.7", "6.3", "5.8", "7.1", "6.3", "6.5", "7.6", "4.9", "7.3", "6.7", "7.2", "6.5", "6.4", "6.8", "5.7", "5.8", "6.4", "6.5", "7.7", "7.7", "6.0", "6.9", "5.6", "7.7", "6.3", "6.7", "7.2", "6.2", "6.1", "6.4", "7.2", "7.4", "7.9", "6.4", "6.3", "6.1", "7.7", "6.3", "6.4", "6.0", "6.9", "6.7", "6.9", "5.8", "6.8", "6.7", "6.7", "6.3", "6.5", "6.2", "5.9"], ["3.5", "3.0", "3.2", "3.1", "3.6", "3.9", "3.4", "3.4", "2.9", "3.1", "3.7", "3.4", "3.0", "3.0", "4.0", "4.4", "3.9", "3.5", "3.8", "3.8", "3.4", "3.7", "3.6", "3.3", "3.4", "3.0", "3.4", "3.5", "3.4", "3.2", "3.1", "3.4", "4.1", "4.2", "3.1", "3.2", "3.5", "3.1", "3.0", "3.4", "3.5", "2.3", "3.2", "3.5", "3.8", "3.0", "3.8", "3.2", "3.7", "3.3", "3.2", "3.2", "3.1", "2.3", "2.8", "2.8", "3.3", "2.4", "2.9", "2.7", "2.0", "3.0", "2.2", "2.9", "2.9", "3.1", "3.0", "2.7", "2.2", "2.5", "3.2", "2.8", "2.5", "2.8",
"2.9", "3.0", "2.8", "3.0", "2.9", "2.6", "2.4", "2.4", "2.7", "2.7", "3.0", "3.4", "3.1", "2.3", "3.0", "2.5", "2.6", "3.0", "2.6", "2.3", "2.7", "3.0", "2.9", "2.9", "2.5", "2.8", "3.3", "2.7", "3.0", "2.9", "3.0", "3.0", "2.5", "2.9", "2.5", "3.6", "3.2", "2.7", "3.0", "2.5", "2.8", "3.2", "3.0", "3.8", "2.6", "2.2", "3.2", "2.8", "2.8", "2.7", "3.3", "3.2", "2.8", "3.0", "2.8", "3.0", "2.8", "3.8", "2.8", "2.8", "2.6", "3.0", "3.4", "3.1", "3.0", "3.1", "3.1", "3.1", "2.7", "3.2", "3.3", "3.0", "2.5", "3.0", "3.4", "3.0"], ["setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa",
"setosa", "setosa", "setosa", "setosa", "setosa", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica",
"virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica"], ["0.2", "0.2", "0.2", "0.2", "0.2", "0.4", "0.3", "0.2", "0.2", "0.1", "0.2", "0.2", "0.1", "0.1", "0.2", "0.4", "0.4", "0.3", "0.3", "0.3", "0.2", "0.4", "0.2", "0.5", "0.2", "0.2", "0.4", "0.2", "0.2", "0.2", "0.2", "0.4", "0.1", "0.2",
"0.1", "0.2", "0.2", "0.1", "0.2", "0.2", "0.3", "0.3", "0.2", "0.6", "0.4", "0.3", "0.2", "0.2", "0.2", "0.2", "1.4", "1.5", "1.5", "1.3", "1.5", "1.3", "1.6", "1.0", "1.3", "1.4", "1.0", "1.5", "1.0", "1.4", "1.3", "1.4", "1.5", "1.0", "1.5", "1.1", "1.8", "1.3", "1.5", "1.2", "1.3", "1.4", "1.4", "1.7", "1.5", "1.0", "1.1", "1.0", "1.2", "1.6", "1.5", "1.6", "1.5", "1.3", "1.3", "1.3", "1.2", "1.4", "1.2", "1.0", "1.3", "1.2", "1.3", "1.3", "1.1", "1.3", "2.5", "1.9", "2.1", "1.8", "2.2", "2.1", "1.7", "1.8", "1.8", "2.5", "2.0", "1.9", "2.1", "2.0", "2.4", "2.3", "1.8", "2.2", "2.3", "1.5", "2.3", "2.0", "2.0", "1.8", "2.1", "1.8", "1.8", "1.8", "2.1", "1.6", "1.9", "2.0", "2.2", "1.5", "1.4", "2.3", "2.4", "1.8", "1.8", "2.1", "2.4", "2.3", "1.9", "2.3", "2.5", "2.3", "1.9", "2.0", "2.3", "1.8"]]
Unique classes are ["setosa", "versicolor", "virginica"]
========================================================================================================================================================
["1.4", "1.4", "1.3", "1.5", "1.4", "1.7", "1.4", "1.5", "1.4", "1.5", "1.5", "1.6", "1.4", "1.1", "1.2", "1.5", "1.3", "1.4", "1.7", "1.5", "1.7", "1.5", "1.0", "1.7", "1.9", "1.6", "1.6", "1.5", "1.4", "1.6", "1.6", "1.5", "1.5", "1.4", "1.5", "1.2", "1.3", "1.5", "1.3", "1.5", "1.3", "1.3", "1.3", "1.6", "1.9", "1.4", "1.6", "1.4", "1.5", "1.4", "4.7", "4.5", "4.9", "4.0", "4.6", "4.5", "4.7", "3.3", "4.6", "3.9", "3.5", "4.2", "4.0", "4.7", "3.6", "4.4", "4.5", "4.1", "4.5", "3.9", "4.8", "4.0", "4.9", "4.7", "4.3", "4.4", "4.8", "5.0", "4.5", "3.5", "3.8", "3.7", "3.9", "5.1", "4.5", "4.5", "4.7", "4.4", "4.1", "4.0", "4.4", "4.6", "4.0", "3.3", "4.2", "4.2", "4.2", "4.3", "3.0", "4.1", "6.0", "5.1", "5.9", "5.6", "5.8", "6.6", "4.5", "6.3", "5.8", "6.1", "5.1", "5.3", "5.5", "5.0", "5.1", "5.3", "5.5",
"6.7", "6.9", "5.0", "5.7", "4.9", "6.7", "4.9", "5.7", "6.0", "4.8", "4.9", "5.6", "5.8", "6.1", "6.4", "5.6", "5.1", "5.6", "6.1", "5.6", "5.5", "4.8", "5.4", "5.6", "5.1", "5.1", "5.9", "5.7", "5.2", "5.0", "5.2", "5.4", "5.1"]
is now
[1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5, 1.7, 1.5, 1.0, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5, 1.4, 1.5, 1.2, 1.3, 1.5, 1.3, 1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.7, 4.5, 4.9, 4.0, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4.0, 4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4.0, 4.9, 4.7, 4.3, 4.4, 4.8, 5.0, 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4.0, 4.4, 4.6, 4.0, 3.3, 4.2, 4.2, 4.2, 4.3, 3.0, 4.1, 6.0, 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3, 5.8, 6.1, 5.1, 5.3, 5.5, 5.0, 5.1, 5.3, 5.5, 6.7, 6.9, 5.0, 5.7, 4.9, 6.7, 4.9, 5.7, 6.0, 4.8, 4.9, 5.6, 5.8, 6.1,
6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 5.5, 4.8, 5.4, 5.6, 5.1, 5.1, 5.9, 5.7, 5.2, 5.0, 5.2, 5.4, 5.1]
with missing values at
[]
========================================================================================================================================================
["setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica"]
is now
["setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica"]
========================================================================================================================================================
Now the unique classes are ["setosa", "versicolor", "virginica"]
========================================================================================================================================================
Categorized ["setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica"] is now [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
========================================================================================================================================================
The value count of SPECIES is {"versicolor": 50, "setosa": 50, "virginica": 50}
========================================================================================================================================================
[1.0, 2.0, 3.0, -5.0, -7.0, 0.0]
In [1.0, 2.0, 3.0, -5.0, -7.0, 0.0]
minimum is -7 and maximum is 3
========================================================================================================================================================
========================================================================================================================================================
[1.0, 2.0, 3.0, -5.0, -7.0, 0.0]
[1.0, 2.0, 3.0, -5.0, -7.0, 0.0]
Normalized to : [0.8, 0.9, 1.0, 0.19999999999999996, 0.0, 0.7]
========================================================================================================================================================
========================================================================================================================================================
[[1, 2], [2, 3], [3, 7], [34, 76]]
[[0.2, 0.3], [0.4, 0.7], [1.0, 2.0], [0.6, 0.0]]
Becomes [[0.6899744811276125, 0.8581489350995123, 0.9933071490757153, 0.6456563062257954], [0.7858349830425586, 0.9478464369215821, 0.9996646498695336, 0.7685247834990175], [0.9370266439430035, 0.9977621514787236, 0.9999999586006244, 0.8581489350995123], [0.9999999999998603, 1.0, 1.0, 0.9999999986183674]]
 using logistic fucntion
========================================================================================================================================================
[1, 2]
 converted to float: [1.0, 2.0]
========================================================================================================================================================
Rounding of 3.14267864 by 4 = 3.1427
========================================================================================================================================================
========================================================================================================================================================
-19.671590217709767
[[1.2, 2.3], [2.4, 3.7], [4.0, 9.0], [4.6, 7.0]]
[[0.8, 1.7], [1.6, 2.3], [2.0, 5.0], [3.4, 5.0]]
[[0.2, 0.6], [0.8, 2.0999999999999996], [3.0, 14.0], [2.4, 6.0]]
========================================================================================================================================================
========================================================================================================================================================
========================================================================================================================================================
========================================================================================================================================================
========================================================================================================================================================
========================================================================================================================================================
========================================================================================================================================================
========================================================================================================================================================
========================================================================================================================================================
========================================================================================================================================================
([[0.8730889653677842, 1.7031193796597321], [4.002650564798213, 6.2047416002724125], [1.0001472545942394, 2.0002878004552267], [0.8015902153859887, 1.3629584974450937]], 3)


Reading the file ...
Number of rows = ~9568
Before removing missing values, number of rows : 9569
After removing missing values, number of rows : 9568
The target here is header named: "PE"
Values are now converted to f64
Train size: 7176
Test size : 2391

The weights of the inputs are [0.314932586684033, 0.037495703481620055, 0.35785467866344134, 0.02960445507944867]
The r2 of this model is23.39456423212376
================================================================================================================================================================
                                                                                "ML OVER"
================================================================================================================================================================
*/
