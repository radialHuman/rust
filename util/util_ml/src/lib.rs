use math::round;
use rand::*;

// use crate::lib_matrix::*;

/*
SOURCE
------
Activation from : https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
Neuron : nnfs

DESCRIPTION
-----------------------------------------
STRUCTS
-------
1. LayerDetails : To create a layer of n_neurons and n_inputs
    > 1. create_weights : To randomly generate n_neurons weights between -1 and 1
    > 2. create_bias : A constant numer (can be modified if required) vector of n_neurons as bias
    > 3. output_of_layer : activation_function((inputs*weights)-bias)

FUNCTIONS
---------
1. activation_leaky_relu :
    > 1. &Vec<T> to be used as input to funtion
    > 2. alpha to control the fucntion's "leaky" nature
    = 1. Modified Vec<T>
2. activation_relu :
    > 1. &Vec<T> to be used as input to funtion
    = 1. Modified Vec<T>
3. activation_sigmoid :
    > 1. &Vec<T> to be used as input to funtion
    = 1. Modified Vec<T>
4. activation_tanh :
    > 1. &Vec<T> to be used as input to funtion
    = 1. Modified Vec<T>
*/
pub struct LayerDetails {
    /*
    To create layers of a neural network
    n_inputs : number of inputs to the layer
    n_neurons : number of neurons in the layer
    */
    pub n_inputs: usize,
    pub n_neurons: i32,
}
impl LayerDetails {
    pub fn create_weights(&self) -> Vec<Vec<f64>> {
        /*
        random weights between -1 and 1, for optimization, assinged to each neuron and input
        */
        let mut rng = rand::thread_rng();
        let mut weight: Vec<Vec<f64>> = vec![];
        // this gives transposed weights
        for _ in 0..self.n_inputs {
            weight.push(
                (0..self.n_neurons)
                    .map(|_| round::ceil(rng.gen_range(-1., 1.), 3))
                    .collect(),
            );
        }
        weight
    }
    pub fn create_bias(&self, value: f64) -> Vec<f64> {
        /*
        Initialize a constant value vector of value passed
        Which acts as bias introduced to each neuron of the layer
        */
        let bias = vec![value; self.n_neurons as usize];
        bias
    }
    pub fn output_of_layer(
        &self,
        input: &Vec<Vec<f64>>,
        weights: &Vec<Vec<f64>>,
        bias: &mut Vec<f64>,
        f: &str,
        alpha: f64,
    ) -> Vec<Vec<f64>> {
        /*
        The inputs are :
        INPUT : [NxM]
        WEIGHTS : [MxN]
        BIAS : [N]
        F: "relu" or "leaky relu" or "sigmoid" or "tanh"
        ALPHA : only if leaky relu is used, else it will be ignored

        The output is [NxN] : F((INPUT*WEIGHTS)+BIAS)
         */
        let mut mat_mul = transpose(&matrix_multiplication(&input, &weights));
        // println!("input * weights = {:?}", mat_mul);
        let mut output: Vec<Vec<f64>> = vec![];
        for i in &mut mat_mul {
            // println!("i*w {:?}, bias {:?}", &i, &bias);
            output.push(vector_addition(i, bias));
        }
        // println!("Before activation it was {:?}", &output[0]);
        // println!("After activation it was {:?}", activation_relu(&output[0]));
        let mut activated_output = vec![];
        match f {
            "relu" => {
                println!("Alpha is for 'leaky relu' only, it is not taken into account here");
                for i in output.clone() {
                    activated_output.push(activation_relu(&i));
                }
            }
            "leaky relu" => {
                for i in output.clone() {
                    activated_output.push(activation_leaky_relu(&i, alpha));
                }
            }
            "sigmoid" => {
                println!("Alpha is for 'leaky relu' only, it is not taken into account here");
                for i in output.clone() {
                    activated_output.push(activation_sigmoid(&i));
                }
            }
            "tanh" => {
                println!("Alpha is for 'leaky relu' only, it is not taken into account here");
                for i in output.clone() {
                    activated_output.push(activation_tanh(&i));
                }
            }
            _ => panic!("Select from either 'tanh','sigmoid','relu','leaky relu'"),
        }
        // transpose(&activated_output)
        activated_output
    }
}

pub fn activation_relu<T>(input: &Vec<T>) -> Vec<T>
where
    T: Copy + std::cmp::PartialOrd + std::ops::Sub<Output = T> + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    // ReLU for neurons
    /*
    If greater than 0 then x passed else 0
    where x is the values of (input*weights)+bias
    */
    let zero = "0".parse::<T>().unwrap();
    input
        .iter()
        .map(|x| if *x > zero { *x } else { *x - *x })
        .collect()
}

pub fn activation_leaky_relu<T>(input: &Vec<T>, alpha: f64) -> Vec<T>
where
    T: Copy + std::cmp::PartialOrd + std::ops::Mul<Output = T> + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    // Leaky ReLU for neurons, where alpha is multiplied with x if x <= 0
    // to avoid making it completely 0 like in ReLU
    /*
    If greater than 0 then x passed else alpha*x
    where x is the values of (input*weights)+bias
    */
    let zero = "0".parse::<T>().unwrap();
    let a = format!("{}", alpha).parse::<T>().unwrap();
    input
        .iter()
        .map(|x| if *x > zero { *x } else { a * *x })
        .collect()
}

pub fn activation_sigmoid<T>(input: &Vec<T>) -> Vec<f64>
where
    T: std::str::FromStr + std::fmt::Debug,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    // Sigmoid for neurons
    /*
    1/(1+(e^x))
    where x is the values of (input*weights)+bias
    */
    input
        .iter()
        .map(|x| 1. / (1. + format!("{:?}", x).parse::<f64>().unwrap().exp()))
        .collect()
}

pub fn activation_tanh<T>(input: &Vec<T>) -> Vec<f64>
where
    T: std::str::FromStr + std::fmt::Debug,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    // TanH for neurons
    input
        .iter()
        .map(|x| {
            (format!("{:?}", x).parse::<f64>().unwrap().exp()
                - (format!("{:?}", x).parse::<f64>().unwrap() * (-1.)).exp())
                / (format!("{:?}", x).parse::<f64>().unwrap().exp()
                    + (format!("{:?}", x).parse::<f64>().unwrap() * (-1.)).exp())
        })
        .collect()
}

/*
DESCRIPTION
-----------------------------------------
STRUCTS
-------
1. OLS : file_path: String, target: usize, // target column number , pub test_size: f64
    > fit

2. BLR : file_path: String, test_size: f64, target_column: usize, learning_rate: f64, iter_count: u32, binary_threshold: f64,
    > fit
    > sigmoid
    > log_loss
    > gradient_descent
    > change_in_loss
    > predict

3. KNN : file_path: String, test_size: f64, target_column: usize, k: usize, method: &'a str
    > fit
    x predict

4. Distance : row1: Vec<f64>, row2: Vec<f64>
    > distance_euclidean
    > distance_manhattan
    > distance_cosine
    > distance_chebyshev

5. Kmeans : file_path: String, k: usize, iterations: u32
    > fit

6. SSVM : file_path: String,  drop_column_number: Vec<usize>,test_size: f64,  learning_rate: f64,  iter_count: i32,  reg_strength: f64
    > fit
    x sgd
    x compute_cost
    x calculate_cost_gradient
    x predict


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

11. min_max_scaler : between [0.,1.]
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

14. logistic_predict :
    1. > matrix1: &Vec<Vec<f64>>
    2. > beta: &Vec<Vec<f64>>
    = Vec<Vec<f64>>

15. randomize_vector :
    1. > rows : &Vec<T>
    = Vec<T>

16. randomize :
    1. > rows : &Vec<Vec<T>>
    = Vec<Vec<T>>

17. train_test_split_vector_f :
    1. > input: &Vec<f64>
    2. > percentage: f64
    = Vec<f64>
    = Vec<f64>

18. train_test_split_f :
    1. > input: &Vec<Vec<f64>>
    2. > percentage: f64
    = Vec<Vec<f64>>
    = Vec<Vec<f64>>

19. correlation :
    1. > list1: &Vec<T>
    2. > list2: &Vec<T>
    3. > name: &str // 's' spearman, 'p': pearson
    = f64

20. std_dev :
    1. > list1: &Vec<T>
    = f64

21. spearman_rank : Spearman ranking
    1. > list1: &Vec<T>
    = Vec<(T, f64)>

22. how_many_and_where_vector :
    1. > list: &Vec<T>
    2. > number: T  // to be searched
    = Vec<usize>

23. how_many_and_where :
    1. > list: &Vec<Vec<T>>
    2. > number: T  // to be searched
    = Vec<(usize,usize)>

24. z_score :
    1. > list: &Vec<T>
    2. > number: T
    = f64

25. one_hot_encoding :
    1. > column: &Vec<&str>
     = Vec<Vec<u8>>

26. shape : shows #rowsx#columns
    1. m: &Vec<Vec<f64>>
    = ()

27. rmse
    1. test_data: &Vec<Vec<f64>>
    2. predicted: &Vec<f64>)
    = f64

28. mse
    1. test_data: &Vec<Vec<f64>>
    2. predicted: &Vec<f64>
    = f64

29. mae
    1. test_data: &Vec<Vec<f64>>
    2. predicted: &Vec<f64>
    = f64

30. r_square
    1. predicted: &Vec<f64>
    2. actual: &Vec<f64>, features: usize
    = (f64, f64)

31. mape
    1. test_data: &Vec<Vec<f64>>
    2. predicted: &Vec<f64>
    = f64

32. drop_column
    1. matrix: &Vec<Vec<f64>>
    2. column_number
    = Vec<Vec<f64>>

33. preprocess_train_test_split
    1. matrix: &Vec<Vec<f64>>,
    2. test_percentage: f64,
    3 .target_column: usize,
    4 .preprocess: &str : "s","m"
     = (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) : x_train, y_train, x_test, y_test

34. standardize_vector_f
    1. list: &Vec<f64>
     = Vec<f64>

35. min_max_scaler
    1. list: &Vec<f64>
    = Vec<f64>

36. float_randomize
    1. matrix: &Vec<Vec<String>>
    = Vec<Vec<f64>>

37. confuse_me
    1. predicted: &Vec<f64>
    2. actual: &Vec<f64>
     = ()

38. cv
    1. data : &Vec<Vec<T>>
    2. k : usize
    = (Vec<Vec<T>>,Vec<Vec<T>>)

39. z_outlier_f
    1. list : &Vec<f64>
    = Vec<f64>

40. percentile_f
    1. list:&Vec<f64>
    2. percentile:u32)
    = f64

41. quartile_f
    1. list:&Vec<f64>

*/

// use crate::lib_matrix;
// use lib_matrix::*;

pub struct OLS {
    pub file_path: String,
    pub target: usize, // target column number
    pub test_size: f64,
}

impl OLS {
    pub fn fit(&self) {
        /*
        Source:
        Video: https://www.youtube.com/watch?v=K_EH2abOp00
        Book: Trevor Hastie,  Robert Tibshirani, Jerome Friedman - The Elements of  Statistical Learning_  Data Mining, Inference, and Pred
        Article: https://towardsdatascience.com/regression-an-explanation-of-regression-metrics-and-what-can-go-wrong-a39a9793d914#:~:text=Root%20Mean%20Squared%20Error%3A%20RMSE,value%20predicted%20by%20the%20model.&text=Mean%20Absolute%20Error%3A%20MAE%20is,value%20predicted%20by%20the%20model.
        Library:

        TODO:
        * Whats the role of gradient descent in this?
        * rules of regression
        * p-value
        * Colinearity
        */

        // read a csv file
        let (columns, values) = read_csv(self.file_path.clone()); // output is row wise
                                                                  // assuming the last column has the value to be predicted
        println!(
            "The target here is header named: {:?}",
            columns[self.target - 1]
        );

        // // converting vector of string to vector of f64s
        let random_data = randomize(&values)
            .iter()
            .map(|a| {
                a.iter()
                    .filter(|b| **b != "".to_string())
                    .map(|b| b.parse::<f64>().unwrap())
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();
        // splitting it into train and test as per test percentage passed as parameter to get scores
        let (train_data, test_data) = train_test_split_f(&random_data, self.test_size);
        shape("Training data", &train_data);
        shape("Testing data", &test_data);

        // converting rows to vector of columns of f64s

        // println!("{:?}",train_data );
        shape("Training data", &train_data);
        let actual_train = row_to_columns_conversion(&train_data);
        // println!(">>>>>");
        let x = drop_column(&actual_train, self.target);
        // // the read columns are in transposed form already, so creating vector of features X and adding 1 in front of it for b0
        let b0_vec: Vec<Vec<f64>> = vec![vec![1.; x[0].len()]]; //[1,1,1...1,1,1]
        let X = [&b0_vec[..], &x[..]].concat(); // [1,1,1...,1,1,1]+X
                                                // shape(&X);
        let xt = MatrixF { matrix: X };

        // and vector of targets y
        let y = vec![actual_train[self.target - 1].to_vec()];
        // print_a_matrix(
        //     "Features",
        //     &xt.matrix.iter().map(|a| a[..6].to_vec()).collect(),
        // );
        // print_a_matrix("Target", &y);

        /*
        beta = np.linalg.inv(X.T@X)@(X.T@y)
         */

        // (X.T@X)
        let xtx = MatrixF {
            matrix: matrix_multiplication(&xt.matrix, &transpose(&xt.matrix)),
        };
        // println!("{:?}", MatrixF::inverse_f(&xtx));
        let slopes = &matrix_multiplication(
            &MatrixF::inverse_f(&xtx), // np.linalg.inv(X.T@X)
            &transpose(&vec![matrix_vector_product_f(&xt.matrix, &y[0])]), //(X.T@y)
        )[0];

        // combining column names with coefficients
        let output: Vec<_> = columns[..columns.len() - 1]
            .iter()
            .zip(slopes[1..].iter())
            .collect();
        // println!("****************** Without Gradient Descent ******************");
        println!(
        "\n\nThe coeficients of a columns as per simple linear regression on {:?}% of data is : \n{:?} and b0 is : {:?}",
        self.test_size * 100.,
        output,
        slopes[0]
    );

        // predicting the values for test features
        // multiplying each test feture row with corresponding slopes to predict the dependent variable
        let mut predicted_values = vec![];
        for i in test_data.iter() {
            predicted_values.push({
                let value = i
                    .iter()
                    .zip(slopes[1..].iter())
                    .map(|(a, b)| (a * b))
                    .collect::<Vec<f64>>();
                value.iter().fold(slopes[0], |a, b| a + b) // b0+b1x1+b2x2..+bnxn
            });
        }

        println!("RMSE : {:?}", rmse(&test_data, &predicted_values));
        println!("MSE : {:?}", mse(&test_data, &predicted_values)); // cost function
        println!("MAE : {:?}", mae(&test_data, &predicted_values));
        println!("MAPE : {:?}", mape(&test_data, &predicted_values));
        println!(
            "R2 and adjusted R2 : {:?}",
            r_square(
                &test_data
                    .iter()
                    .map(|a| a[test_data[0].len() - 1])
                    .collect(), // passing only the target values
                &predicted_values,
                columns.len(),
            )
        );

        println!();
        println!();
    }
}

pub struct BLR {
    pub file_path: String,     // pointing to a csv or txt file
    pub test_size: f64,        // ex: .30 => random 30% of data become test
    pub target_column: usize,  // column index which has to be classified
    pub learning_rate: f64,    // gradient descent step size ex: 0.1, 0.05 etc
    pub iter_count: u32,       // how many epochs ex: 10000
    pub binary_threshold: f64, // at what probability will the class be determined ex: 0.6 => anything above 0.6 is 1
}
impl BLR {
    pub fn fit(&self) {
        /*
            Source:
            Video:
            Book: Trevor Hastie,  Robert Tibshirani, Jerome Friedman - The Elements of  Statistical Learning_  Data Mining, Inference, and Pred
            Article: https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
            Library:
        */

        // read a csv file
        let (_, values) = read_csv(self.file_path.clone()); // output is row wise

        // converting vector of string to vector of f64s
        let random_data = float_randomize(&values);

        // splitting it into train and test as per test percentage passed as parameter to get scores
        let (x_train, y_train, x_test, y_test) =
            preprocess_train_test_split(&random_data, self.test_size, self.target_column, "");

        shape("Training features", &x_train);
        shape("Test features", &x_test);
        println!("Training target: {:?}", &y_train.len());
        println!("Test target: {:?}", &y_test.len());

        // now to the main part
        let length = x_train[0].len();
        let feature_count = x_train.len();
        // let class_count = (unique_values(&y_test).len() + unique_values(&y_test).len()) / 2;
        let intercept = vec![vec![1.; length]];
        let new_x_train = [&intercept[..], &x_train[..]].concat();
        let mut coefficients = vec![0.; feature_count + 1];

        let mut cost = vec![];
        print!("Reducing loss ...");
        for _ in 0..self.iter_count {
            let s = BLR::sigmoid(&new_x_train, &coefficients);
            cost.push(BLR::log_loss(&s, &y_train));
            let gd = BLR::gradient_descent(&new_x_train, &s, &y_train);
            coefficients = BLR::change_in_loss(&coefficients, self.learning_rate, &gd);
        }
        // println!("The intercept is : {:?}", coefficients[0]);
        // println!(
        //     "The coefficients are : {:?}",
        //     columns
        //         .iter()
        //         .zip(coefficients[1..].to_vec())
        //         .collect::<Vec<(&String, f64)>>()
        // );
        let predicted = BLR::predict(&x_test, &coefficients, self.binary_threshold);
        confuse_me(&predicted, &y_test, -1., 1.);
    }

    pub fn predict(test_features: &Vec<Vec<f64>>, weights: &Vec<f64>, threshold: f64) -> Vec<f64> {
        let length = test_features[0].len();
        let intercept = vec![vec![1.; length]];
        let new_x_test = [&intercept[..], &test_features[..]].concat();
        let pred = BLR::sigmoid(&new_x_test, weights);
        pred.iter()
            .map(|a| if *a > threshold { 1. } else { 0. })
            .collect()
    }

    pub fn change_in_loss(coeff: &Vec<f64>, lr: f64, gd: &Vec<f64>) -> Vec<f64> {
        print!(".");
        if coeff.len() == gd.len() {
            element_wise_operation(coeff, &gd.iter().map(|a| a * lr).collect(), "add")
        } else {
            panic!("The dimensions do not match")
        }
    }

    pub fn gradient_descent(
        train: &Vec<Vec<f64>>,
        sigmoid: &Vec<f64>,
        y_train: &Vec<f64>,
    ) -> Vec<f64> {
        let part2 = element_wise_operation(sigmoid, y_train, "sub");
        let numerator = matrix_vector_product_f(train, &part2);
        numerator
            .iter()
            .map(|a| *a / (y_train.len() as f64))
            .collect()
    }

    pub fn log_loss(sigmoid: &Vec<f64>, y_train: &Vec<f64>) -> f64 {
        let part11 = sigmoid.iter().map(|a| a.log(1.0_f64.exp())).collect();
        let part12 = y_train.iter().map(|a| a * -1.).collect();
        let part21 = sigmoid
            .iter()
            .map(|a| (1. - a).log(1.0_f64.exp()))
            .collect();
        let part22 = y_train.iter().map(|a| 1. - a).collect();
        let part1 = element_wise_operation(&part11, &part12, "mul");
        let part2 = element_wise_operation(&part21, &part22, "mul");
        mean(&element_wise_operation(&part1, &part2, "sub"))
    }

    pub fn sigmoid(train: &Vec<Vec<f64>>, coeff: &Vec<f64>) -> Vec<f64> {
        let z = matrix_vector_product_f(&transpose(train), coeff);
        z.iter().map(|a| 1. / (1. + a.exp())).collect()
    }
}

pub struct KNN<'a> {
    pub file_path: String,
    pub test_size: f64,
    pub target_column: usize,
    pub k: usize,
    pub method: &'a str,
}
impl<'a> KNN<'a> {
    pub fn fit(&self) {
        /*
        method : Euclidean or Chebyshev or cosine or Manhattan or Minkowski or Weighted
        */
        // read a csv file
        let (_, values) = read_csv(self.file_path.clone()); // output is row wise

        // converting vector of string to vector of f64s
        let random_data = float_randomize(&values); // blr needs to be removed

        // splitting it into train and test as per test percentage passed as parameter to get scores
        let (x_train, y_train, x_test, y_test) =
            preprocess_train_test_split(&random_data, self.test_size, self.target_column, ""); // blr needs to be removed

        // now to the main part
        // since it is row wise, conversion
        let train_rows = columns_to_rows_conversion(&x_train);
        let test_rows = columns_to_rows_conversion(&x_test);
        shape("train Rows:", &train_rows);
        shape("test Rows:", &test_rows);
        // println!("{:?}", y_train.len());

        // predicting values
        let predcited = KNN::predict(&train_rows, &y_train, &test_rows, self.method, self.k);
        println!("Metrics");
        confuse_me(
            &predcited.iter().map(|a| *a as f64).collect::<Vec<f64>>(),
            &y_test,
            -1.,
            1.,
        ); // blr needs to be removed
    }

    fn predict(
        train_rows: &Vec<Vec<f64>>,
        train_values: &Vec<f64>,
        test_rows: &Vec<Vec<f64>>,
        method: &str,
        k: usize,
    ) -> Vec<i32> {
        match method {
            "e" => println!("\n\nCalculating KNN using euclidean distance ..."),
            "ma" => println!("\n\nCalculating KNN using manhattan distance ..."),
            "co" => println!("\n\nCalculating KNN using cosine distance ..."),
            "ch" => println!("\n\nCalculating KNN using chebyshev distance ..."),
            _ => panic!("The method has to be either 'e' or 'ma' or 'co' or 'ch'"),
        };
        let mut predcited = vec![];
        for j in test_rows.iter() {
            let mut class_found = vec![];
            for (n, i) in train_rows.iter().enumerate() {
                // println!("{:?},{:?},{:?}", j, n, i);
                let dis = Distance {
                    row1: i.clone(),
                    row2: j.clone(),
                };
                match method {
                    "e" => class_found.push((dis.distance_euclidean(), train_values[n])),
                    "ma" => class_found.push((dis.distance_manhattan(), train_values[n])),
                    "co" => class_found.push((dis.distance_cosine(), train_values[n])),
                    "ch" => class_found.push((dis.distance_chebyshev(), train_values[n])),
                    _ => (), // cant happen as it would panic in the previous match
                };
            }
            // sorting acsending the vector by first value of tuple
            class_found.sort_by(|(a, _), (c, _)| (*a).partial_cmp(c).unwrap());
            let k_nearest = class_found[..k].to_vec();
            let knn: Vec<f64> = k_nearest.iter().map(|a| a.1).collect();
            // converting classes to int and classifying
            let nearness = value_counts(&knn.iter().map(|a| *a as i32).collect());
            // finding the closest
            predcited.push(*nearness.iter().next_back().unwrap().0)
        }
        predcited
    }
}

pub struct Distance {
    pub row1: Vec<f64>,
    pub row2: Vec<f64>,
}
impl Distance {
    pub fn distance_euclidean(&self) -> f64 {
        // sqrt(sum((row1-row2)**2))

        let distance = self
            .row1
            .iter()
            .zip(self.row2.iter())
            .map(|(a, b)| (*a - *b) * (*a - *b))
            .collect::<Vec<f64>>();
        distance.iter().fold(0., |a, b| a + b).sqrt()
    }

    pub fn distance_manhattan(&self) -> f64 {
        // sum(|row1-row2|)

        let distance = self
            .row1
            .iter()
            .zip(self.row2.iter())
            .map(|(a, b)| (*a - *b).abs())
            .collect::<Vec<f64>>();
        distance.iter().fold(0., |a, b| a + b)
    }

    pub fn distance_cosine(&self) -> f64 {
        // 1- (a.b)/(|a||b|)

        let numerator = self
            .row1
            .iter()
            .zip(self.row2.iter())
            .map(|(a, b)| (*a * *b))
            .collect::<Vec<f64>>()
            .iter()
            .fold(0., |a, b| a + b);
        let denominator = (self
            .row1
            .iter()
            .map(|a| a * a)
            .collect::<Vec<f64>>()
            .iter()
            .fold(0., |a, b| a + b)
            .sqrt())
            * (self
                .row2
                .iter()
                .map(|a| a * a)
                .collect::<Vec<f64>>()
                .iter()
                .fold(0., |a, b| a + b)
                .sqrt());
        1. - numerator / denominator
    }

    pub fn distance_chebyshev(&self) -> f64 {
        // max(|row1-row2|)
        let distance = self
            .row1
            .iter()
            .zip(self.row2.iter())
            .map(|(a, b)| (*a - *b).abs())
            .collect::<Vec<f64>>();
        distance.iter().cloned().fold(0. / 0., f64::max)
    }
}

pub struct Kmeans {
    pub file_path: String,
    pub k: usize,
    pub iterations: u32,
}
impl Kmeans {
    pub fn fit(&self) {
        /*
            Source:
            Video:
            Book: Trevor Hastie,  Robert Tibshirani, Jerome Friedman - The Elements of  Statistical Learning_  Data Mining, Inference, and Pred
            Article: https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/
            Library:

            ABOUT:
            * Assuming no duplicate rows exist
            * Only features and no targets in the input data

            Procedure:
            1. Prepare data : remove target if any
            2. Select K centroids
            3. Find closest points (Eucledian distance)
            4. Calcualte new mean
            5. Repeat 3,4 till the same points ened up in the cluster

            TODO:
            * Add cost function to minimize
        */

        // read a csv file
        let (_, values) = read_csv(self.file_path.clone()); // output is row wise

        // converting vector of string to vector of f64s
        let random_data: Vec<_> = float_randomize(&values);

        // selecting first k points as centroid (already in random order)
        let mut centroids = randomize(&random_data)[..self.k].to_vec();
        print_a_matrix("Original means", &centroids);

        let mut new_mean: Vec<Vec<f64>> = vec![];
        for x in 0..self.iterations - 1 {
            let mut updated_cluster = vec![];
            let mut nearest_centroid_number = vec![];
            for i in random_data.iter() {
                let mut distance = vec![];
                for (centroid_number, j) in centroids.iter().enumerate() {
                    let dis = Distance {
                        row1: i.clone(),
                        row2: j.clone(),
                    };
                    distance.push((centroid_number, dis.distance_euclidean()))
                }
                distance.sort_by(|m, n| m.1.partial_cmp(&n.1).unwrap());
                nearest_centroid_number.push(distance[0].0);
            }

            // combining cluster number and data
            let clusters: Vec<(&usize, &Vec<f64>)> = nearest_centroid_number
                .iter()
                .zip(random_data.iter())
                .collect();
            // println!("{:?}", clusters);

            // finding new centorid
            new_mean = vec![];
            for (m, _) in centroids.iter().enumerate() {
                let mut group = vec![];
                for i in clusters.iter() {
                    if *i.0 == m {
                        group.push(i.1.clone());
                    }
                }
                new_mean.push(
                    group
                        .iter()
                        .fold(vec![0.; self.k], |a, b| {
                            element_wise_operation(&a, b, "add")
                        })
                        .iter()
                        .map(|a| a / (group.len() as f64)) // the mean part in K-means
                        .collect(),
                );
                updated_cluster = clusters.clone()
            }
            println!("Iteration {:?}", x);
            if centroids == new_mean {
                // show in a list of cluster number as per the order of row in original data
                let mut rearranged_output = vec![];
                for i in values
                    .iter()
                    .map(|a| a.iter().map(|b| b.parse().unwrap()).collect())
                    .collect::<Vec<Vec<f64>>>()
                    .iter()
                {
                    for (c, v) in updated_cluster.iter() {
                        if i == *v {
                            rearranged_output.push((c, v));
                            break;
                        }
                    }
                }
                // displaying only the clusters assigned to  each row
                println!(
                    "CLUSTERS\n{:?}",
                    rearranged_output
                        .iter()
                        .map(|a| **(a.0))
                        .collect::<Vec<usize>>()
                );
                break;
            } else {
                centroids = new_mean.clone();
            }
        }
        print_a_matrix("Final means", &centroids);
    }
}

pub struct SSVM {
    pub file_path: String,              // pointing to a csv or txt file
    pub drop_column_number: Vec<usize>, // if first and second column has id and are not required then vec![1,2] else if nothing then vec![]
    pub test_size: f64,                 // ex: .30 => random 30% of data become test
    pub learning_rate: f64,             // gradient descent step size ex: 0.1, 0.05 etc
    pub iter_count: i32,                // how many epochs ex: 10000
    pub reg_strength: f64, // at what probability will the class be determined ex: 0.6 => anything above 0.6 is 1
}
impl SSVM {
    // https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2
    // data: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data (changed M:1 and B:2)
    /*
        Assumes all the pre-processing like converting string columns to a number has been done
        Assuming the target column is placed in the end
        Assuming only two classes 1 and -1
    */

    pub fn fit(&self) -> Vec<f64> {
        // read a csv file
        let (columns, values) = read_csv(self.file_path.clone()); // output is row wise

        // converting vector of string to vector of f64s
        // println!("___");
        let mut random_data = SSVM::float_randomize(&values);

        println!(
            "The columns are\n{:?}\n",
            columns
                .iter()
                .filter(|a| **a != "\r".to_string())
                .map(|a| a.replace("\"", ""))
                .collect::<Vec<String>>()
        );
        shape("Before dropping columns the dimensions are", &random_data);

        // drop column after converting it to column wise data
        random_data = row_to_columns_conversion(&random_data);
        if self.drop_column_number.len() > 0 {
            for (n, i) in self.drop_column_number.iter().enumerate() {
                if n == 0 {
                    println!("Dropping column #{}", i);
                    random_data = drop_column(&random_data, *i);
                } else {
                    println!("Dropping column #{}", i);
                    random_data = drop_column(&random_data, *i - n);
                }
            }
        }
        // converting it back to row wise
        random_data = columns_to_rows_conversion(&random_data);

        shape("After dropping columns the dimensions are", &random_data);
        println!();

        head(&random_data, 5);

        // normalizing features in thier columns wise format
        let mut normalized = row_to_columns_conversion(&random_data);

        random_data = row_to_columns_conversion(&random_data);
        for (n, i) in random_data.iter().enumerate() {
            print!(".");
            if n != normalized.len() - 1 - self.drop_column_number.len() {
                normalized[n] = min_max_scaler(i);
            } else {
                normalized[n] = i.clone();
            }
        }
        println!("\nAfter normalizing:");

        // converting it back to row wise
        normalized = columns_to_rows_conversion(&normalized);
        head(&normalized, 5);
        println!();

        // splitting it into train and test as per test percentage passed as parameter to get scores
        let (mut x_train, y_train, mut x_test, y_test) =
            preprocess_train_test_split(&normalized, self.test_size, normalized[0].len(), "");

        // adding intercept column to feature
        let mut length = x_train[0].len();
        let intercept = vec![vec![1.; length]];
        x_train = [&intercept[..], &x_train[..]].concat();
        length = x_test[0].len();
        let intercept = vec![vec![1.; length]];
        x_test = [&intercept[..], &x_test[..]].concat();

        // converting into proper shape
        x_train = columns_to_rows_conversion(&x_train);
        x_test = columns_to_rows_conversion(&x_test);

        // checking the shapes
        shape("Training features", &x_train);
        shape("Test features", &x_test);
        println!("Training target: {:?}", &y_train.len());
        println!("Test target: {:?}", &y_test.len());

        let weights = SSVM::sgd(&self, &x_train, &y_train);
        let predictions = SSVM::predict(&self, &x_test, &weights);
        confuse_me(&predictions, &y_test, -1., 1.);
        println!("Weights of intercept followed by features : {:?}", weights);
        weights
    }
    fn sgd(&self, features: &Vec<Vec<f64>>, output: &Vec<f64>) -> Vec<f64> {
        let max_epoch: i32 = self.iter_count;
        let mut weights = vec![0.; features[0].len()];
        let mut nth = 0.;
        let mut prev_cost = std::f64::INFINITY;
        let per_cost_threshold = 0.01;
        for epoch in 1..max_epoch {
            // shuffling inputs
            if epoch % 100 == 0 {
                print!("..");
            }
            let order = randomize_vector(&(0..output.len()).map(|a| a).collect());
            let mut x = vec![];
            let mut y = vec![];
            for i in order.iter() {
                x.push(features[*i].clone());
                y.push(output[*i]);
            }

            // calculating cost
            for (n, i) in x.iter().enumerate() {
                let ascent = SSVM::calculate_cost_gradient(&self, &weights, i, y[n]);
                weights = element_wise_operation(
                    &weights,
                    &ascent.iter().map(|a| a * self.learning_rate).collect(),
                    "sub",
                );
            }
            // println!("Ascent {:?}", weights);

            if epoch == 2f64.powf(nth) as i32 || epoch == max_epoch - 1 {
                let cost = SSVM::compute_cost(&self, &weights, features, output);
                println!("{} Epoch, has cost {}", epoch, cost);
                if (prev_cost - cost).abs() < (per_cost_threshold * prev_cost) {
                    println!("{:?}", weights);
                    return weights;
                }
                prev_cost = cost;
                nth += 1.;
            }
        }
        // println!();
        weights
    }

    fn compute_cost(&self, weight: &Vec<f64>, x: &Vec<Vec<f64>>, y: &Vec<f64>) -> f64 {
        // hinge loss
        let mut distance = element_wise_operation(&matrix_vector_product_f(x, weight), &y, "mul");
        // println!("{:?}", &matrix_vector_product_f(x, weight).len());
        // println!("Loss {:?}", distance);
        distance = distance.iter().map(|a| 1. - *a).collect();
        distance = distance
            .iter()
            .map(|a| if *a > 0. { *a } else { 0. })
            .collect();
        let hinge_loss =
            self.reg_strength * (distance.iter().fold(0., |a, b| a + b) / (x.len() as f64));
        (dot_product(&weight, &weight) / 2.) + hinge_loss
    }

    fn calculate_cost_gradient(
        &self,
        weight: &Vec<f64>,
        x_batch: &Vec<f64>,
        y_batch: f64,
    ) -> Vec<f64> {
        let distance = 1. - (dot_product(&x_batch, &weight) * y_batch);
        // println!("Distance {:?}", distance);
        let mut dw = vec![0.; weight.len()];
        let di;
        if distance < 0. {
            di = dw.clone();
        } else {
            let second_half = x_batch
                .iter()
                .map(|a| a * self.reg_strength * y_batch)
                .collect();
            di = element_wise_operation(weight, &second_half, "sub");
        }
        dw = element_wise_operation(&di, &dw, "add");
        // println!("di : {:?}", dw);
        dw
    }

    fn predict(&self, test_features: &Vec<Vec<f64>>, weights: &Vec<f64>) -> Vec<f64> {
        let mut output = vec![];
        for i in test_features.iter() {
            if dot_product(i, weights) > 0. {
                output.push(1.);
            } else {
                output.push(-1.);
            }
        }
        println!("Predications : {:?}", output);
        output
    }

    fn float_randomize(matrix: &Vec<Vec<String>>) -> Vec<Vec<f64>> {
        randomize(
            &matrix
                .iter()
                .map(|a| {
                    a.iter()
                        .map(|b| {
                            (b).replace("\r", "")
                                .replace("\n", "")
                                .parse::<f64>()
                                .unwrap()
                        })
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>(),
        )
    }
}

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
    // https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/
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
    // https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/
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
    // https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/
    let mu1 = mean(list1);
    let mu2 = mean(list2);
    let zero: T = "0".parse().unwrap();
    let _len_str: f64 = list1.len().to_string().parse().unwrap(); // if division is required
    let tupled: Vec<_> = list1.iter().zip(list2).collect();
    let output = tupled.iter().fold(zero, |a, b| {
        a + ((*b.0 - mu1.to_string().parse().unwrap()) * (*b.1 - mu2.to_string().parse().unwrap()))
    });
    let numerator: f64 = output.to_string().parse().unwrap();
    numerator // / _len_str  // (this is not being divided by populaiton size)
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
    /*
    To find slope and intercept of a line
    */
    // https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/
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
    // https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/
    let train_features = &train.iter().map(|a| a.0).collect();
    let test_features = &test.iter().map(|a| a.1).collect();
    let (offset, slope) = coefficient(train_features, test_features);
    let b0: T = offset.to_string().parse().unwrap();
    let b1: T = slope.to_string().parse().unwrap();
    let predicted_output = test.iter().map(|a| b0 + b1 * a.0).collect();
    let original_output: Vec<_> = test.iter().map(|a| a.0).collect();
    println!("========================================================================================================================================================");
    println!("b0 = {:?} and b1= {:?}", b0, b1);
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
    // https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/
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
    /*
    Returns headers and row wise values as strings
    */
    // println!("========================================================================================================================================================");
    println!("Reading the file ...");
    let file = fs::read_to_string(&path).unwrap();
    let splitted: Vec<&str> = file.split("\n").collect();
    let rows: i32 = (splitted.len() - 1) as i32;
    println!("Number of rows = {}", rows);
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
    /*
    Convert a vector to a type by passing a value of that type and pass a value to replace missing values
    */
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
    /*
    Replace missing value with the string thats passed
    */
    // println!("========================================================================================================================================================");
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

pub fn min_max_scaler(list: &Vec<f64>) -> Vec<f64> {
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

    let first_calc_t = transpose(&first_calc);
    let mut x = vec![];
    for j in 0..matrix1[0].len() {
        let mut row = vec![];
        for i in matrix1.iter() {
            row.push(i[j]);
        }
        x.push(row);
    }

    // PYTHON : // final_calc = np.dot(first_calc.T, x)
    let mut final_calc = vec![];
    for i in first_calc_t.iter() {
        for j in x.iter() {
            final_calc.push(dot_product(&i, &j))
        }
    }

    // println!("{:?}\n{:?}", &first_calc_t, &x);
    // println!("{:?}", &final_calc);
    // println!(
    //     "{:?}",
    //     shape_changer(&final_calc, matrix1[0].len(), matrix1.len())
    // );
    shape_changer(&final_calc, matrix1[0].len(), matrix1.len())
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

pub fn randomize_vector<T: std::clone::Clone>(rows: &Vec<T>) -> Vec<T> {
    /*
    Shuffle values inside vector
    */
    use rand::seq::SliceRandom;
    // use rand::thread_rng;
    let mut order: Vec<usize> = (0..rows.len() as usize).collect();
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

pub fn randomize<T: std::clone::Clone>(rows: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    /*
    Shuffle rows inside matrix
    */
    use rand::seq::SliceRandom;
    // use rand::thread_rng;
    let mut order: Vec<usize> = (0..rows.len() as usize).collect();
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

pub fn train_test_split_vector_f(input: &Vec<f64>, percentage: f64) -> (Vec<f64>, Vec<f64>) {
    /*
    Shuffle and split percentage of test for vector
    */
    // shuffle
    let data = randomize_vector(input);
    // println!("{:?}", data);
    // split
    let test_count = (data.len() as f64 * percentage) as usize;
    // println!("Test size is {:?}", test_count);

    let test = data[0..test_count].to_vec();
    let train = data[test_count..].to_vec();
    (train, test)
}

pub fn train_test_split_f(
    input: &Vec<Vec<f64>>,
    percentage: f64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    /*
    Shuffle and split percentage of test for matrix
    */
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

pub fn correlation<T>(list1: &Vec<T>, list2: &Vec<T>, name: &str) -> f64
where
    T: std::iter::Sum<T>
        + std::ops::Div<Output = T>
        + std::fmt::Debug
        + std::fmt::Display
        + std::ops::Add
        + std::cmp::PartialOrd
        + std::marker::Copy
        + std::ops::Add<T, Output = T>
        + std::ops::Sub<T, Output = T>
        + std::ops::Mul<T, Output = T>
        + std::string::ToString
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    /*
    Correlation
    "p" => pearson
    "s" => spearman's
    */
    let cov = covariance(list1, list2);
    let output = match name {
        "p" => (cov / (std_dev(list1) * std_dev(list2))) / list1.len() as f64,
        "s" => {
            // https://statistics.laerd.com/statistical-guides/spearmans-rank-order-correlation-statistical-guide-2.php
            //covariance(&rank(list1), &rank(list2))/(std_dev(&rank(list1))*std_dev(&rank(list2)))
            let ranked_list1 = spearman_rank(list1);
            let ranked_list2 = spearman_rank(list2);
            let len = list1.len() as f64;
            // sorting rnaks back to original positions
            let mut rl1 = vec![];
            for k in list1.iter() {
                for (i, j) in ranked_list1.iter() {
                    if k == i {
                        rl1.push(j);
                    }
                }
            }
            let mut rl2 = vec![];
            for k in list2.iter() {
                for (i, j) in ranked_list2.iter() {
                    if k == i {
                        rl2.push(j);
                    }
                }
            }

            let combined: Vec<_> = rl1.iter().zip(rl2.iter()).collect();
            let sum_of_square_of_difference = combined
                .iter()
                .map(|(a, b)| (***a - ***b) * (***a - ***b))
                .fold(0., |a, b| a + b);
            1. - ((6. * sum_of_square_of_difference) / (len * ((len * len) - 1.)))
            // 0.
        }
        _ => panic!("Either `p`: Pearson or `s`:Spearman has to be the name. Please retry!"),
    };
    match output {
        x if x < 0.2 && x > -0.2 => println!("There is a weak correlation between the two :"),
        x if x > 0.6 => println!("There is a strong positive correlation between the two :"),
        x if x < -0.6 => println!("There is a strong negative correlation between the two :"),
        _ => (),
    }
    output
}

pub fn std_dev<T>(list1: &Vec<T>) -> f64
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
    let mu: T = mean(list1).to_string().parse().unwrap();
    let square_of_difference = list1.iter().map(|a| (*a - mu) * (*a - mu)).collect();
    let var = mean(&square_of_difference);
    var.sqrt()
}

pub fn spearman_rank<T>(list1: &Vec<T>) -> Vec<(T, f64)>
where
    T: std::iter::Sum<T>
        + std::ops::Div<Output = T>
        + std::fmt::Debug
        + std::fmt::Display
        + std::ops::Add
        + std::marker::Copy
        + std::cmp::PartialOrd
        + std::ops::Add<T, Output = T>
        + std::ops::Sub<T, Output = T>
        + std::ops::Mul<T, Output = T>
        + std::string::ToString
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    /*
    Returns ranking of each value in ascending order with thier spearman rank in a vector of tuple
    */
    // https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    let mut sorted = list1.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut map: Vec<(_, _)> = vec![];
    for (n, i) in sorted.iter().enumerate() {
        map.push(((n + 1), *i));
    }
    // repeating values
    let mut repeats: Vec<_> = vec![];
    for (n, i) in sorted.iter().enumerate() {
        if how_many_and_where_vector(&sorted, *i).len() > 1 {
            repeats.push((*i, how_many_and_where_vector(&sorted, *i)));
        } else {
            repeats.push((*i, vec![n]));
        }
    }
    // calculating the rank
    let rank: Vec<_> = repeats
        .iter()
        .map(|(a, b)| {
            (a, b.iter().fold(0., |a, b| a + *b as f64) / b.len() as f64) // mean of each position vector
        })
        .collect();
    let output: Vec<_> = rank.iter().map(|(a, b)| (**a, b + 1.)).collect(); // 1. is fro index offset
    output
}

pub fn how_many_and_where_vector<T>(list: &Vec<T>, number: T) -> Vec<usize>
where
    T: std::cmp::PartialEq + std::fmt::Debug + Copy,
{
    /*
    Returns the positions of the number to be found in a vector
    */
    let tuple: Vec<_> = list
        .iter()
        .enumerate()
        .filter(|&(_, a)| *a == number)
        .map(|(n, _)| n)
        .collect();
    tuple
}

pub fn how_many_and_where<T>(matrix: &Vec<Vec<T>>, number: T) -> Vec<(usize, usize)>
where
    T: std::cmp::PartialEq + std::fmt::Debug + Copy,
{
    /*
    Returns the positions of the number to be found in a matrix
    */
    let mut output = vec![];
    for (n, i) in matrix.iter().enumerate() {
        for j in how_many_and_where_vector(&i, number) {
            output.push((n, j));
        }
    }
    output
}

pub fn z_score<T>(list: &Vec<T>, number: T) -> f64
where
    T: std::iter::Sum<T>
        + std::ops::Div<Output = T>
        + Copy
        + std::str::FromStr
        + std::string::ToString
        + std::ops::Add<T, Output = T>
        + std::ops::Sub<T, Output = T>
        + std::ops::Mul<T, Output = T>
        + std::fmt::Debug
        + std::cmp::PartialEq
        + std::fmt::Display
        + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    /*
    Returns z_score
    */
    let n: f64 = number.to_string().parse().unwrap();
    if list.contains(&number) {
        (n - mean(list)) / std_dev(list)
    } else {
        panic!("The number not found in vector passed, please check");
    }
}

pub fn one_hot_encoding(column: &Vec<&str>) -> Vec<Vec<u8>> {
    /*
    Counts unique values
    creates those many new columns
    each column will have 1 for every occurance of a particular unique value
    Ex: ["A", "B", "C"] => [[1,0,0],[0,1,0],[0,0,1]]
    */
    let values = unique_values(&column.clone());
    // println!("{:?}", values);
    let mut output = vec![];
    for i in values.iter() {
        output.push(column.iter().map(|a| if a == i { 1 } else { 0 }).collect());
    }
    output
}

pub fn shape(words: &str, m: &Vec<Vec<f64>>) {
    // # of rows and columns of a matrix
    println!(
        "{:?} : Rows: {:?}, Columns: {:?}",
        words,
        m.len(),
        m[0].len()
    );
}

pub fn rmse(test_data: &Vec<Vec<f64>>, predicted: &Vec<f64>) -> f64 {
    /*
    square root of (square of difference of predicted and actual divided by number of predications)
    */
    (mse(test_data, predicted)).sqrt()
}

pub fn mse(test_data: &Vec<Vec<f64>>, predicted: &Vec<f64>) -> f64 {
    /*
    square of difference of predicted and actual divided by number of predications
    */

    let mut square_error: Vec<f64> = vec![];
    for (n, i) in test_data.iter().enumerate() {
        let j = match i.last() {
            Some(x) => (predicted[n] - x) * (predicted[n] - x), // square difference
            _ => panic!("Something wrong in passed test data"),
        };
        square_error.push(j)
    }
    // println!("{:?}", square_error);
    square_error.iter().fold(0., |a, b| a + b) / (predicted.len() as f64)
}

pub fn mae(test_data: &Vec<Vec<f64>>, predicted: &Vec<f64>) -> f64 {
    /*
    average of absolute difference of predicted and actual
    */

    let mut absolute_error: Vec<f64> = vec![];
    for (n, i) in test_data.iter().enumerate() {
        let j = match i.last() {
            Some(x) => (predicted[n] - x).abs(), // absolute difference
            _ => panic!("Something wrong in passed test data"),
        };
        absolute_error.push(j)
    }
    // println!("{:?}", absolute_error);
    absolute_error.iter().fold(0., |a, b| a + b) / (predicted.len() as f64)
}

pub fn r_square(predicted: &Vec<f64>, actual: &Vec<f64>, features: usize) -> (f64, f64) {
    // https://github.com/radialHuman/rust/blob/master/util/util_ml/src/lib_ml.rs
    /*

    */
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
    let degree_of_freedom = predicted.len() as f64 - 1. - features as f64;
    let ar2 = 1. - ((1. - r2) * ((predicted.len() as f64 - 1.) / degree_of_freedom));
    (r2, ar2)
}

pub fn mape(test_data: &Vec<Vec<f64>>, predicted: &Vec<f64>) -> f64 {
    /*
    average of absolute difference of predicted and actual
    */

    let mut absolute_error: Vec<f64> = vec![];
    for (n, i) in test_data.iter().enumerate() {
        let j = match i.last() {
            Some(x) => (((predicted[n] - x) / predicted[n]).abs()) * 100., // absolute difference
            _ => panic!("Something wrong in passed test data"),
        };
        absolute_error.push(j)
    }
    // println!("{:?}", absolute_error);
    absolute_error.iter().fold(0., |a, b| a + b) / (predicted.len() as f64)
}

pub fn drop_column(matrix: &Vec<Vec<f64>>, column_number: usize) -> Vec<Vec<f64>> {
    // let part1 = matrix[..column_number - 1].to_vec();
    // let part2 = matrix[column_number..].to_vec();
    // shape("target", &part2);
    [
        &matrix[..column_number - 1].to_vec()[..],
        &matrix[column_number..].to_vec()[..],
    ]
    .concat()
}

pub fn float_randomize(matrix: &Vec<Vec<String>>) -> Vec<Vec<f64>> {
    randomize(
        &matrix
            .iter()
            .map(|a| {
                a.iter()
                    .map(|b| (*b).replace("\r", "").parse::<f64>().unwrap())
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>(),
    )
}

pub fn preprocess_train_test_split(
    matrix: &Vec<Vec<f64>>,
    test_percentage: f64,
    target_column: usize,
    preprocess: &str,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    /*
    preprocess : "s" : standardize, "m" : minmaxscaler, "_" : no change
    */

    let (train_data, test_data) = train_test_split_f(matrix, test_percentage);
    // println!("Training size: {:?}", train_data.len());
    // println!("Test size: {:?}", test_data.len());

    // converting rows to vector of columns of f64s
    let mut actual_train = row_to_columns_conversion(&train_data);
    let mut actual_test = row_to_columns_conversion(&test_data);

    match preprocess {
        "s" => {
            actual_train = actual_train
                .iter()
                .map(|a| standardize_vector_f(a))
                .collect::<Vec<Vec<f64>>>();
            actual_test = actual_test
                .iter()
                .map(|a| standardize_vector_f(a))
                .collect::<Vec<Vec<f64>>>();
        }
        "m" => {
            actual_train = actual_train
                .iter()
                .map(|a| min_max_scaler(a))
                .collect::<Vec<Vec<f64>>>();
            actual_test = actual_test
                .iter()
                .map(|a| min_max_scaler(a))
                .collect::<Vec<Vec<f64>>>();
        }

        _ => println!("Using the actual values without preprocessing unless 's' or 'm' is passed"),
    };

    (
        drop_column(&actual_train, target_column),
        actual_train[target_column - 1].clone(),
        drop_column(&actual_test, target_column),
        actual_test[target_column - 1].clone(),
    )
}

pub fn standardize_vector_f(list: &Vec<f64>) -> Vec<f64> {
    /*
    Preserves the shape of the original distribution. Doesn't
    reduce the importance of outliers. Least disruptive to the
    information in the original data. Default range for
    MinMaxScaler is O to 1.
        */
    list.iter()
        .map(|a| (*a - mean(list)) / std_dev(list))
        .collect()
}

// pub fn min_max_scaler(list: &Vec<f64>) -> Vec<f64> {
//     let (minimum, maximum) = min_max_f(&list);
//     let range: f64 = maximum - minimum;
//     list.iter().map(|a| 1. - ((maximum - a) / range)).collect()
// }

pub fn confuse_me(predicted: &Vec<f64>, actual: &Vec<f64>, class0: f64, class1: f64) {
    // https://medium.com/@MohammedS/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b
    let mut tp = 0.; // class_one_is_class_one
    let mut fp = 0.; // class_one_is_class_two(Type 1)
    let mut fng = 0.; // class_two_is_class_one (Type 1)
    let mut tng = 0.; // class_two_is_class_two

    for (i, j) in actual
        .iter()
        .zip(predicted.iter())
        .collect::<Vec<(&f64, &f64)>>()
        .iter()
    {
        if **i == class0 && **j == class0 {
            tp += 1.;
        }
        if **i == class1 && **j == class1 {
            tng += 1.;
        }
        if **i == class0 && **j == class1 {
            fp += 1.;
        }
        if **i == class1 && **j == class0 {
            fng += 1.;
        }
    }
    println!("\n|------------------------|");
    println!("|  {:?}    |   {:?}", tp, fp);
    println!("|------------------------|");
    println!("|  {:?}    |   {:?}", fng, tng);
    println!("|------------------------|");
    println!("Accuracy : {:.3}", (tp + tng) / (tp + fp + fng + tng));
    println!("Precision : {:.3}", (tp) / (tp + fp));
    let precision: f64 = (tp) / (tp + fp);
    println!("Recall (sensitivity) : {:.3}", (tp) / (tp + fng));
    let recall: f64 = (tp) / (tp + fng);
    println!("Specificity: {:.3}", (tng) / (fp + tng));
    println!(
        "F1 : {:.3}\n\n",
        (2. * precision * recall) / (precision * recall)
    );
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

pub fn z_outlier_f(list: &Vec<f64>) -> Vec<f64> {
    /*
    Anything below -3 or beyond 3 std deviations is considered as an outlier
    */

    let mut v_clone = list.clone();
    v_clone.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let z_v: Vec<_> = v_clone
        .iter()
        .map(|a| (z_score(&v_clone, *a), *a))
        .collect();
    z_v.iter()
        .filter(|(a, _)| (*a > 3.) || (*a < -3.))
        .map(|a| a.1)
        .collect::<Vec<f64>>()
}

pub fn percentile_f(list: &Vec<f64>, percentile: u32) -> f64 {
    /*
    Returns passed percentile in the list
    */
    // https://en.wikipedia.org/wiki/Percentile
    list.clone().sort_by(|a, b| a.partial_cmp(b).unwrap());
    let oridinal_rank = round_off_f((percentile as f64 / 100.) * (list.len() as f64), 0);
    list[oridinal_rank as usize - 1]
}

pub fn quartile_f(list: &Vec<f64>) {
    /*
    Returns quartiles like in a boxplot
    */
    println!(
        "\tPercentile:\t10th :{:?}\t25th :{:?}\t50th :{:?}\t75th :{:?}\t90th :{:?}",
        percentile_f(list, 10),
        percentile_f(list, 25),
        percentile_f(list, 50),
        percentile_f(list, 75),
        percentile_f(list, 90)
    );
}

/*
DESCRIPTION
-----------------------------------------
STRUCTS
-------
1. MatrixF : matrix: Vec<Vec<f64>> // upto 100x100
    > determinant_f
    > inverse_f
    > is_square_matrix
    x round_off_f

2. DataFrame : string: Vec<Vec<&'a str>>, numbers: Vec<Vec<f64>>, boolean: Vec<Vec<bool>>,
    > groupby : string_column_number , operation: &str // "sum" or "mean"
    > describe
    > sort : col_type : "s" or "n" , col_number, ascending : true or false

2. DataMap : string: HashMap<&'a str,Vec<&'a str>>, numbers:HashMap<&'a str,Vec<f64>>, boolean: HashMap<&'a str,Vec<bool>>,
    > groupby : string_column_number , operation: &str // "sum" or "mean"
    > describe
    > sort : col_type : "s" or "n" , col_name, ascending : true or false

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

27. datamap_comparision
    > table1: &DataMap
    > table2: &DataMap

28. dataframe_comparision
    > table1: &DataFrame
    > table2: &DataFrame

29. compare_vectors
    > v1: &Vec<T>,
    > v2: &Vec<T>,
    = (usize, Vec<(usize, usize)>)


*/

#[derive(Debug)] // to make it usable by print!
pub struct MatrixF {
    pub matrix: Vec<Vec<f64>>,
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
                    "100".parse().unwrap()
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

    fn round_off_f(value: f64, decimals: i32) -> f64 {
        // println!("========================================================================================================================================================");
        ((value * 10.0f64.powi(decimals)).round()) / 10.0f64.powi(decimals)
    }

    pub fn inverse_f(&self) -> Vec<Vec<f64>> {
        // https://integratedmlai.com/matrixinverse/
        let mut input = self.matrix.clone();
        let length = self.matrix.len();
        let mut identity = MatrixF::identity_matrix(length);

        let index: Vec<usize> = (0..length).collect();
        // let int_index: Vec<i32> = index.iter().map(|a| *a as i32).collect();

        for diagonal in 0..length {
            let diagonal_scalar = 1. / (input[diagonal][diagonal]);
            // first action
            for column_loop in 0..length {
                input[diagonal][column_loop] *= diagonal_scalar;
                identity[diagonal][column_loop] *= diagonal_scalar;
            }

            // second action
            let except_diagonal: Vec<usize> = index[0..diagonal]
                .iter()
                .copied()
                .chain(index[diagonal + 1..].iter().copied())
                .collect();
            // println!("Here\n{:?}", exceptDiagonal);

            for i in except_diagonal {
                let row_scalar = input[i as usize][diagonal].clone();
                for j in 0..length {
                    input[i][j] = input[i][j] - (row_scalar * input[diagonal][j]);
                    identity[i][j] = identity[i][j] - (row_scalar * identity[diagonal][j])
                }
            }
        }

        identity
    }

    fn identity_matrix(size: usize) -> Vec<Vec<f64>> {
        let mut output: Vec<Vec<f64>> = MatrixF::zero_matrix(size);
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

    fn zero_matrix(size: usize) -> Vec<Vec<f64>> {
        let mut output: Vec<Vec<f64>> = vec![];
        for _ in 0..=(size - 1) {
            output.push(vec![0.; size]);
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
        println!("      Details of the DataFrame",);
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
            "Grouped on {:?} => {:?}",
            string_column_number,
            unique_string
                .iter()
                .zip(output.iter())
                .collect::<Vec<(&&str, &f64)>>()
        );
    }
    pub fn sort(&self, col_type: &str, col_number: usize, ascending: bool) -> DataFrame {
        /* returns a different DataFrame with rows sorted as per order passed
        col_type : "s": string,"n":numerical
        */
        let mut output = DataFrame {
            string: vec![],
            numerical: vec![],
            boolean: vec![],
        };
        let mut to_sort_by_string;
        let mut to_sort_by_numerical;
        let order: Vec<usize>;
        match col_type {
            "s" => {
                to_sort_by_string = self.string[col_number].clone();
                // finding the order of sorting
                order = DataFrame::find_order_of_sorting_string(&mut to_sort_by_string, ascending);
            }
            "n" => {
                to_sort_by_numerical = self.numerical[col_number].clone();
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
        for each_vector in self.string.iter() {
            let mut new_vector = vec![];
            for o in order.iter() {
                new_vector.push(each_vector[*o]);
            }
            output.string.push(new_vector);
        }

        // reordering the original DataFrame (Numerical)
        for each_vector in self.numerical.iter() {
            let mut new_vector = vec![];
            for o in order.iter() {
                new_vector.push(each_vector[*o]);
            }

            output.numerical.push(new_vector);
        }

        // reordering the original DataFrame (Boolean)
        for each_vector in self.boolean.iter() {
            let mut new_vector = vec![];
            for o in order.iter() {
                new_vector.push(each_vector[*o]);
            }
            output.boolean.push(new_vector);
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

pub struct DataMap<'a> {
    // use std::collections::HashMap;
    // stored column wise
    pub string: HashMap<&'a str, Vec<&'a str>>,
    pub numerical: HashMap<&'a str, Vec<f64>>,
    pub boolean: HashMap<&'a str, Vec<bool>>,
}
impl<'a> DataMap<'a> {
    pub fn describe(&self) {
        println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        println!("      Details of the DataMap",);
        println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        for (k, v) in self.string.iter() {
            println!(
                "String column :{:?}  Values count : {:?}",
                k,
                value_counts(v)
            )
        }
        for (k, v) in self.boolean.iter() {
            println!(
                "Boolean column :{:?}  Values count : {:?}",
                k,
                value_counts(v)
            )
        }
        for (k, v) in self.numerical.iter() {
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
        let reduced_dataframe_float: Vec<&Vec<f64>> = self.numerical.values().clone().collect();

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
            "Grouped by {:?} => {:?}",
            string_column,
            unique_string
                .iter()
                .zip(output.iter())
                .collect::<Vec<(&&str, &f64)>>()
        );
    }
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
            let new_a = pad_with_zero(a, b.len() - a.len(), "post");
            println!("The changed vector is {:?}", new_a);
            for i in 0..a.len() {
                output.push(a[i] + b[i]);
            }
            output
        } else {
            let new_b = pad_with_zero(b, a.len() - b.len(), "post");
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
    println!(
        "Multiplication of {}x{} and {}x{}",
        input.len(),
        input[0].len(),
        weights.len(),
        weights[0].len()
    );
    println!("Output will be {}x{}", input.len(), weights[0].len());
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
    /*
    operations between two vectors, by passing paramters: "mul","sub","div","add"
    */
    if a.len() == b.len() {
        a.iter().zip(b.iter()).map(|(x, y)| match operation {
                        "mul" => *x * *y,
                        "add" => *x + *y,
                        "sub" => *x - *y,
                        "div" => *x / *y,
                        _ => panic!("Operation unsuccessful!\nEnter any of the following(case sensitive):\n> Add\n> Sub\n> Mul\n> Div"),
                    })
                    .collect()
    } else {
        panic!("Dimension mismatch")
    }
}

pub fn pad_with_zero<T>(vector: &mut Vec<T>, count: usize, position: &str) -> Vec<T>
where
    T: Copy + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    /*
    Prefixing or postfixing 0s of a vector
    position : `post` or `pre`
    */
    let mut output = vector.clone();
    let zero = "0".parse::<T>().unwrap();
    match position {
        "post" => {
            for _ in 0..count {
                output.push(zero);
            }
        }
        "pre" => {
            let z = vec![zero; count];
            output = [&z[..], &vector[..]].concat()
        }
        _ => panic!("Position can either be `post` or `pre`"),
    };
    output
}

pub fn make_matrix_float<T>(input: &Vec<Vec<T>>) -> Vec<Vec<f64>>
where
    T: std::fmt::Display + Copy,
{
    /*
    Convert each element of matrix into f64
    */
    // println!("========================================================================================================================================================");
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
    /*
    Convert each element of vector into f64
    */
    // println!("========================================================================================================================================================");
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
    /*
    round off a f64 to the number decimals passed
    */
    // println!("========================================================================================================================================================");
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
        if negative.len() > 0 && positive.len() > 0 {
            (negative[0], positive[positive.len() - 1])
        } else {
            if positive.len() == 0 && negative.len() != 0 {
                (negative[negative.len() - 1], negative[0])
            } else {
                if negative.len() == 0 && positive.len() != 0 {
                    (positive[0], positive[positive.len() - 1])
                } else {
                    panic!("Empty vector found")
                }
            }
        }
    } else {
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

// use std::collections::BTreeMap;
pub fn value_counts<T: std::cmp::Ord>(list: &Vec<T>) -> BTreeMap<T, u32>
where
    T: std::cmp::PartialEq + std::cmp::Eq + std::hash::Hash + Copy,
{
    /*
    Returns a dictioanry of every unique value with its frequency count
    */
    // println!("========================================================================================================================================================");
    let mut count: BTreeMap<T, u32> = BTreeMap::new();
    for i in list {
        count.insert(*i, 1 + if count.contains_key(i) { count[i] } else { 0 });
    }
    count
}

use std::any::type_name;
pub fn type_of<T>(_: T) -> &'static str {
    /*
    Returns the type of data passed
    */
    type_name::<T>()
}

pub fn unique_values<T>(list: &Vec<T>) -> Vec<T>
where
    T: std::cmp::PartialEq + Copy,
{
    /*
    Reruns a set of distinct values for the vector passed
    */
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
    /*
    Similar to elemenet wise vector operations, by passing paramters: "mul","sub","div","add"
    */
    if matrix1.len() == matrix2.len() && matrix1[0].len() == matrix2[0].len() {
        matrix1
            .iter()
            .zip(matrix2.iter())
            .map(|(x, y)| {
                x.iter()
                    .zip(y.iter())
                    .map(|a| match operation {
                        "mul" => *a.0 * *a.1,
                        "add" => *a.0 + *a.1,
                        "sub" => *a.0 - *a.1,
                        "div" => *a.0 / *a.1,
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
    /*
    Dot product of each row of matrix with a vector
    */
    let mut output: Vec<_> = vec![];
    if matrix[0].len() == vector.len() {
        for i in matrix.iter() {
            output.push(dot_product(i, vector));
        }
    } else {
        panic!("The lengths do not match, please check");
    }
    output
}

pub fn split_vector<T: std::clone::Clone>(vector: &Vec<T>, parts: i32) -> Vec<Vec<T>> {
    /*
    Breaks vector into multiple parts if length is divisible by the # of parts
    */
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
        panic!("This partition is not possible, check the number of partitions passed")
    }
}

pub fn split_vector_at<T>(vector: &Vec<T>, at: T) -> Vec<Vec<T>>
where
    T: std::cmp::PartialEq + Copy + std::clone::Clone,
{
    /*
    Splits a vector into 2 at if a particular value is found
    */
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
    let a = matrix1;
    let b = matrix2;
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

pub fn head<T: std::clone::Clone + std::fmt::Debug>(data: &Vec<Vec<T>>, rows: usize) {
    /*
    Works on row wise data
    Shows first few rows of a matrix
    */
    if rows <= data.len() {
        let output = data[..rows].to_vec();
        print_a_matrix(&format!("First {} rows", rows), &output);
    } else {
        panic!("Data is nt that big, please check the numbers");
    }
}

pub fn tail<T: std::clone::Clone + std::fmt::Debug>(data: &Vec<Vec<T>>, rows: usize) {
    /*
    Works on row wise data
    Shows first few rows of a matrix
    */
    if rows <= data.len() {
        let output = data[data.len() - rows..].to_vec();
        print_a_matrix(&format!("Last {} rows", rows), &output);
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
    let mut output: Vec<Vec<_>> = vec![];
    for j in 0..(data[0].len()) {
        let columns = data.iter().map(|a| a[j]).collect();
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

pub fn datamap_comparision(table1: &DataMap, table2: &DataMap) {
    /*
    Generates report on count, similarities and dissimilarities of 2 DataMaps
    */

    // comparing column count
    println!("\n********** Count comparision **********");
    let string_columns1 = table1.string.keys().collect::<Vec<&&str>>();
    let string_columns2 = table2.string.keys().collect::<Vec<&&str>>();

    if string_columns1.len() == string_columns2.len() {
        println!("Number of String columns match");
    } else {
        println!(
            "Mismatch in count of String columns : Table 1 has {} ; while Table 2 has {:?}",
            string_columns1.len(),
            string_columns2.len()
        );
    }

    let numerical_columns1 = table1.numerical.keys().collect::<Vec<&&str>>();
    let numerical_columns2 = table2.numerical.keys().collect::<Vec<&&str>>();

    if numerical_columns1.len() == numerical_columns2.len() {
        println!("Number of Numerical columns match");
    } else {
        println!(
            "Mismatch in count of Numerical columns : Table 1 has {} while Table 2 has {:?}",
            numerical_columns1.len(),
            numerical_columns2.len()
        );
    }

    let boolean_columns1 = table1.boolean.keys().collect::<Vec<&&str>>();
    let boolean_columns2 = table2.boolean.keys().collect::<Vec<&&str>>();

    if boolean_columns1.len() == boolean_columns2.len() {
        println!("Number of Boolean columns match");
    } else {
        println!(
            "Mismatch in count of Boolean columns : Table 1 has {} while Table 2 has {:?}",
            boolean_columns1.len(),
            boolean_columns2.len()
        );
    }

    println!("\n********** Column name comparision **********");
    // chekcing for duplicate headers d=is not required as hashmap takes care of that automatically

    let mut c = 0;
    let mut mis_string1 = vec![];
    let mut mis_string2 = vec![];
    for i in string_columns1.iter() {
        if string_columns2.contains(i) {
            c += 1;
        } else {
            mis_string2.push(i);
        }
    }

    for i in string_columns2.iter() {
        if string_columns1.contains(i) {
            c += 1;
        } else {
            mis_string1.push(i);
        }
    }
    // println!("{:?}", c);
    if c == string_columns1.len() + string_columns2.len() {
        println!("String columns match (irrespective of order)");
    } else {
        if mis_string1.len() > 0 && mis_string2.len() > 0 {
            println!(
                "Table 1 has {:?} missing in String ; Table 2 has {:?} missing in String",
                mis_string1, mis_string2
            );
        }
        if mis_string1.len() > 0 && mis_string2.len() == 0 {
            println!("Table 1 has {:?} missing in String", mis_string1);
        }
        if mis_string1.len() == 0 && mis_string2.len() > 0 {
            println!("Table 2 has {:?} missing in String", mis_string2);
        }
    }

    c = 0;
    let mut mis_numerical1 = vec![];
    let mut mis_numerical2 = vec![];
    for i in numerical_columns1.iter() {
        if numerical_columns2.contains(i) {
            c += 1;
        } else {
            mis_numerical2.push(i);
        }
    }

    for i in numerical_columns2.iter() {
        if numerical_columns1.contains(i) {
            c += 1;
        } else {
            mis_numerical1.push(i);
        }
    }
    // println!("{:?}", c);
    if c == numerical_columns1.len() + numerical_columns2.len() {
        println!("Numerical columns match (irrespective of order)");
    } else {
        if mis_numerical1.len() > 0 && mis_numerical2.len() > 0 {
            println!(
                "Table 1 has {:?} missing in Numerical ; Table 2 has {:?} missing in Numerical",
                mis_numerical1, mis_numerical2
            );
        }
        if mis_numerical1.len() > 0 && mis_numerical2.len() == 0 {
            println!("Table 1 has {:?} missing in Numerical", mis_numerical1);
        }
        if mis_numerical1.len() == 0 && mis_numerical2.len() > 0 {
            println!("Table 2 has {:?} missing in Numerical", mis_numerical2);
        }
    }

    c = 0;
    let mut mis_boolean1 = vec![];
    let mut mis_boolean2 = vec![];
    for i in boolean_columns1.iter() {
        if boolean_columns2.contains(i) {
            c += 1;
        } else {
            mis_boolean2.push(i);
        }
    }

    for i in boolean_columns2.iter() {
        if boolean_columns1.contains(i) {
            c += 1;
        } else {
            mis_boolean1.push(i);
        }
    }
    // println!("{:?}", c);
    if c == boolean_columns1.len() + boolean_columns2.len() {
        println!("Boolean columns match (irrespective of order)");
    } else {
        if mis_boolean1.len() > 0 && mis_boolean2.len() > 0 {
            println!(
                "Table 1 has {:?} missing in Boolean ; Table 2 has {:?} missing in Boolean",
                mis_boolean1, mis_boolean2
            );
        }
        if mis_boolean1.len() > 0 && mis_boolean2.len() == 0 {
            println!("Table 1 has {:?} missing in Boolean", mis_boolean1);
        }
        if mis_boolean1.len() == 0 && mis_boolean2.len() > 0 {
            println!("Table 2 has {:?} missing in Boolean", mis_boolean2);
        }
    }

    println!("\n********** Value comparision (for the common columns) **********");
    let mut string_similarity = 0;
    let mut dissimilarity = vec![];
    for (k1, v1) in table1.string.iter() {
        for (k2, v2) in table2.string.iter() {
            if k1 == k2 {
                string_similarity += compare_vectors(&v1, &v2).0;
                dissimilarity.push(compare_vectors(&v1, &v2).1);
            }
        }
    }
    if string_similarity == table1.string.len() {
        println!("The string values matchs, if present");
    } else {
        println!("Dissimilar in String at (Table 1, Table 2): ");
        let _ = dissimilarity
            .iter()
            .enumerate()
            .map(|(n, a)| {
                println!(
                    "{:?} : {:?}",
                    table1.string.keys().collect::<Vec<&&str>>()[n],
                    a
                );
                a.clone()
            })
            .collect::<Vec<Vec<(usize, usize)>>>();
    }

    let mut numerical_similarity = 0;
    dissimilarity = vec![];
    for (k1, v1) in table1.numerical.iter() {
        for (k2, v2) in table2.numerical.iter() {
            if k1 == k2 {
                numerical_similarity += compare_vectors(&v1, &v2).0;
                dissimilarity.push(compare_vectors(&v1, &v2).1);
            }
        }
    }
    if numerical_similarity == table1.numerical.len() {
        println!("The numerical values matchs, if present");
    } else {
        println!("Dissimilar in Numerical at (Table 1, Table 2): ");
        let _ = dissimilarity
            .iter()
            .enumerate()
            .map(|(n, a)| {
                println!(
                    "{:?} : {:?}",
                    table1.numerical.keys().collect::<Vec<&&str>>()[n],
                    a
                );
                a.clone()
            })
            .collect::<Vec<Vec<(usize, usize)>>>();
    }

    let mut boolean_similarity = 0;
    dissimilarity = vec![];
    for (k1, v1) in table1.boolean.iter() {
        for (k2, v2) in table2.boolean.iter() {
            if k1 == k2 {
                boolean_similarity += compare_vectors(&v1, &v2).0;
                dissimilarity.push(compare_vectors(&v1, &v2).1);
            }
        }
    }
    if boolean_similarity == table1.boolean.len() {
        println!("The Boolean values matchs, if present");
    } else {
        println!("Dissimilar in boolean at (Table 1, Table 2): ");
        let _ = dissimilarity
            .iter()
            .enumerate()
            .map(|(n, a)| {
                println!(
                    "{:?} : {:?}",
                    table1.boolean.keys().collect::<Vec<&&str>>()[n],
                    a
                );
                a.clone()
            })
            .collect::<Vec<Vec<(usize, usize)>>>();
    }
}

pub fn dataframe_comparision(table1: &DataFrame, table2: &DataFrame) {
    /*
    Generates report on count, similarities and dissimilarities of 2 DataMaps
    */
    // comparing column count
    println!("\n********** Count comparision **********");
    if table1.string.len() == table2.string.len() {
        println!("String columns count : {:?}", table1.string.len(),);
    } else {
        println!(
            "String columns count are not the same {:?} and {:?}",
            table1.string.len(),
            table2.string.len()
        );
    }

    if table1.numerical.len() == table2.numerical.len() {
        println!("Numerical columns count : {:?}", table1.numerical.len(),);
    } else {
        println!(
            "Numerical columns count are not the same {:?} and {:?}",
            table1.numerical.len(),
            table2.numerical.len(),
        );
    }

    if table1.boolean.len() == table2.boolean.len() {
        println!("Boolean columns count : {:?}", table1.boolean.len(),);
    } else {
        println!(
            "Boolean columns count are not the same {:?} and {:?}",
            table1.boolean.len(),
            table2.boolean.len()
        );
    }

    println!("\n********** Value comparision (for the common columns) **********");
    let mut string_similarity = 0;
    let mut dissimilarity = vec![];
    for (ni, i) in table1.string.iter().enumerate() {
        for (nj, j) in i.iter().enumerate() {
            for (nk, k) in table2.string.iter().enumerate() {
                for (nl, l) in k.iter().enumerate() {
                    if nj == nl && nk == ni {
                        if j == l {
                            string_similarity += 1;
                        } else {
                            dissimilarity.push(((ni, nj), (nk, nl)));
                        }
                    }
                }
            }
        }
    }
    if string_similarity == table1.string[0].len() * table1.string.len() {
        println!("The string values matchs, if present");
    } else {
        println!("Dissimilar in String at :");
        let _ = dissimilarity
            .iter()
            .enumerate()
            .map(|(n, a)| {
                println!("{:?} : {:?}", n, a);
                *a
            })
            .collect::<Vec<((usize, usize), (usize, usize))>>();
    }

    let mut numerical_similarity = 0;
    let mut dissimilarity = vec![];
    for (ni, i) in table1.numerical.iter().enumerate() {
        for (nj, j) in i.iter().enumerate() {
            for (nk, k) in table2.numerical.iter().enumerate() {
                for (nl, l) in k.iter().enumerate() {
                    if nj == nl && nk == ni {
                        if j == l {
                            numerical_similarity += 1;
                        } else {
                            dissimilarity.push(((ni, nj), (nk, nl)));
                        }
                    }
                }
            }
        }
    }
    if numerical_similarity == table1.numerical[0].len() * table1.numerical.len() {
        println!("The numerical values matchs, if present");
    } else {
        println!("Dissimilar in Numerical at :");
        let _ = dissimilarity
            .iter()
            .enumerate()
            .map(|(n, a)| {
                println!("{:?} : {:?}", n, a);
                *a
            })
            .collect::<Vec<((usize, usize), (usize, usize))>>();
    }

    let mut boolean_similarity = 0;
    let mut dissimilarity = vec![];
    for (ni, i) in table1.boolean.iter().enumerate() {
        for (nj, j) in i.iter().enumerate() {
            for (nk, k) in table2.boolean.iter().enumerate() {
                for (nl, l) in k.iter().enumerate() {
                    if nj == nl && nk == ni {
                        if j == l {
                            boolean_similarity += 1;
                        } else {
                            dissimilarity.push(((ni, nj), (nk, nl)));
                        }
                    }
                }
            }
        }
    }
    if boolean_similarity == table1.boolean[0].len() * table1.boolean.len() {
        println!("The boolean values matchs, if present");
    } else {
        println!("Dissimilar in Boolean at :");
        let _ = dissimilarity
            .iter()
            .enumerate()
            .map(|(n, a)| {
                println!("{:?} : {:?}", n, a);
                *a
            })
            .collect::<Vec<((usize, usize), (usize, usize))>>();
    }
}

pub fn compare_vectors<T: std::cmp::PartialEq>(
    v1: &Vec<T>,
    v2: &Vec<T>,
) -> (usize, Vec<(usize, usize)>) {
    let mut similarity = 0;
    let mut dissimilarity = vec![];
    for (n, i) in v1.iter().enumerate() {
        for (m, j) in v2.iter().enumerate() {
            if n == m {
                if *i == *j {
                    similarity += 1;
                } else {
                    dissimilarity.push((n, m))
                }
            }
        }
    }
    (similarity, dissimilarity)
}

/*
DESCRIPTION
-----------------------------------------
STRUCTS
-------
1. StringToMatch :
        > compare_percentage : comparision based on presence of characters and its position
            x calculate
        > clean_string : lower it and keep alphaneumericals only
            x char_vector
        > compare_chars
        > compare_position
        > fuzzy_subset : scores based on chuncks of string
            x n_gram
        > split_alpha_numericals : seperates numbers from the rest
        > char_count : Returns dictioanry of characters arranged in alphabetically increasing order with their frequency
        > frequent_char : Returns the more frequently occuring character in the string passed
        > char_replace : Finds a character, replaces it with a string at all positions or at just the first depending on operation argument

FUNCTIONS
---------
1. extract_vowels_consonants : Returns a tuple of vectors containing chars (after converting to lowercase)
    > 1. string : String
    = Vec<chars>
    = Vec<chars>

2. sentence_case
    > 1. string : String
    = String

3. remove_stop_words : Based on NLTK removing words that dont convey much from a string
    > 1. string : String
    = String

4. tokenize :
    > string: String
    > symbol: &Vec<&'a str>
     = Vec<String>

*/
use std::collections::BTreeMap;
pub struct StringToMatch {
    pub string1: String,
    pub string2: String,
}

impl StringToMatch {
    pub fn compare_percentage(
        &self,
        weightage_for_position: f64,
        weightage_for_presence: f64,
    ) -> f64 {
        /*
            Scores by comparing characters and its position as per weightage passed
            Weightage passed as ratio
            ex: 2.,1. will give double weightage to position than presence
        */

        ((StringToMatch::compare_chars(&self) * weightage_for_presence * 100.)
            + (StringToMatch::compare_position(&self) * weightage_for_position * 100.))
            / 2.
    }

    pub fn clean_string(s1: String) -> String {
        /*
            Lowercase and removes special characters
        */

        // case uniformity
        let this = s1.to_lowercase();

        // only alpha neurmericals accents - bytes between 48-57 ,97-122, 128-201
        // https://www.utf8-chartable.de/unicode-utf8-table.pl?number=1024&utf8=dec&unicodeinhtml=dec
        let this_byte: Vec<_> = this
            .as_bytes()
            .iter()
            .filter(|a| {
                (**a > 47 && **a < 58) || (**a > 96 && **a < 123) || (**a > 127 && **a < 201)
            })
            .map(|a| *a)
            .collect();
        let new_this = std::str::from_utf8(&this_byte[..]).unwrap();
        new_this.to_string()
    }

    fn char_vector(string1: String) -> Vec<char> {
        /*
            String to vector of characters
        */
        let string1 = StringToMatch::clean_string(string1.clone());
        string1.chars().collect()
    }

    fn calculate(actual: f64, v1: &Vec<char>, v2: &Vec<char>) -> f64 {
        /*
            normalizes score by dividing it with the longest string's length
        */
        let larger = if v1.len() > v2.len() {
            v1.len()
        } else {
            v2.len()
        };
        (actual / larger as f64)
    }

    pub fn compare_chars(&self) -> f64 {
        /*
            Scores as per occurance of characters
        */
        let mut output = 0.;
        // println!("{:?} vs {:?}", self.string1, self.string2);
        let vec1 = StringToMatch::char_vector(self.string1.clone());
        let vec2 = StringToMatch::char_vector(self.string2.clone());

        for i in vec1.iter() {
            if vec2.contains(i) {
                output += 1.;
            }
        }
        StringToMatch::calculate(output, &vec1, &vec2)
    }
    pub fn compare_position(&self) -> f64 {
        /*
            Scores as per similar positioning of characters
        */
        let mut output = 0.;
        // println!("{:?} vs {:?}", self.string1, self.string2);
        let vec1 = StringToMatch::char_vector(self.string1.clone());
        let vec2 = StringToMatch::char_vector(self.string2.clone());

        let combined: Vec<_> = vec1.iter().zip(vec2.iter()).collect();

        for (i, j) in combined.iter() {
            if i == j {
                output += 1.;
            }
        }
        StringToMatch::calculate(output, &vec1, &vec2)
    }

    pub fn fuzzy_subset(&self, n_gram: usize) -> f64 {
        /*
            break into chuncks and compare if not a subset
        */
        let match_percentage;
        let vec1 = StringToMatch::clean_string(self.string1.clone());
        let vec2 = StringToMatch::clean_string(self.string2.clone());

        // finding the subset out of the two parameters
        let mut subset = vec2.clone();
        let mut superset = vec1.clone();
        if vec1.len() < vec2.len() {
            subset = vec1;
            superset = vec2;
        }

        let mut chunck_match_count = 0.;

        // whole string
        if superset.contains(&subset) {
            match_percentage = 100.
        } else {
            // breaking them into continous chuncks
            let superset_n = StringToMatch::n_gram(&superset, n_gram);
            let subset_n = StringToMatch::n_gram(&subset, n_gram);
            for i in subset_n.iter() {
                if superset_n.contains(i) {
                    chunck_match_count += 1.;
                }
            }
            // calculating match score
            let smaller = if superset_n.len() < subset_n.len() {
                superset_n.len()
            } else {
                subset_n.len()
            };
            match_percentage = (chunck_match_count / smaller as f64) * 100.
        }

        println!("{:?} in {:?}", subset, superset);
        match_percentage
    }

    fn n_gram<'a>(string: &'a str, window_size: usize) -> Vec<&'a str> {
        let vector: Vec<_> = string.chars().collect();
        let mut output = vec![];
        for (mut n, _) in vector.iter().enumerate() {
            while n + window_size < string.len() - 1 {
                // println!("Working");
                output.push(&string[n..n + window_size]);
                n = n + window_size;
            }
        }
        unique_values(&output)
    }

    pub fn split_alpha_numericals(string: String) -> (String, String) {
        /*
        "Something 123 else" => ("123","Something  else")
        */
        let bytes: Vec<_> = string.as_bytes().to_vec();
        let numbers: Vec<_> = bytes.iter().filter(|a| **a < 58 && **a > 47).collect();
        println!("{:?}", bytes);
        let aplhabets: Vec<_> = bytes
            .iter()
            .filter(|a| {
                (**a > 64 && **a < 91) // A-Z
                    || (**a > 96 && **a < 123) // a-z
                    || (**a > 127 && **a < 201) // letters with accents
                    || (**a == 32) // spaces
            })
            .collect();

        (
            // to have output as concatenated string
            String::from_utf8(numbers.iter().map(|a| **a).collect()).unwrap(),
            String::from_utf8(aplhabets.iter().map(|a| **a).collect()).unwrap(),
        )
    }

    pub fn char_count(string: String) -> BTreeMap<char, u32> {
        /*
        "SOmething Else" => {' ': 1, 'e': 3, 'g': 1, 'h': 1, 'i': 1, 'l': 1, 'm': 1, 'n': 1, 'o': 1, 's': 2, 't': 1}
         */
        let mut count: BTreeMap<char, Vec<i32>> = BTreeMap::new();
        let vector: Vec<_> = string.to_lowercase().chars().collect();

        // empty dictiornaty
        for i in vector.iter() {
            count.insert(*i, vec![]);
        }
        // dictionary with 1
        let mut new_count: BTreeMap<char, Vec<i32>> = BTreeMap::new();
        for (k, _) in count.iter() {
            let mut values = vec![];
            for i in vector.iter() {
                if i == k {
                    values.push(1);
                }
            }
            new_count.insert(*k, values);
        }

        // dictionary with sum of 1s
        let mut output = BTreeMap::new();
        for (k, v) in new_count.iter() {
            output.insert(*k, v.iter().fold(0, |a, b| a as u32 + *b as u32));
        }

        output
    }

    pub fn frequent_char(string: String) -> char {
        /*
            "SOmething Else" => 'e'
        */
        let dict = StringToMatch::char_count(string);
        let mut value = 0;
        let mut key = '-';
        for (k, _) in dict.iter() {
            key = match dict.get_key_value(k) {
                Some((x, y)) => {
                    if *y > value {
                        value = *y;
                        *x
                    } else {
                        key
                    }
                }
                _ => panic!("Please check the input!!"),
            };
        }
        key
    }

    pub fn char_replace(string: String, find: char, replace: String, operation: &str) -> String {
        /*
        ALL : SOmething Else is now "SOmZthing ElsZ"
        First : SOmething Else is now "SOmZthing Else"
        */

        if string.contains(find) {
            let string_utf8 = string.as_bytes().to_vec();
            let find_utf8 = find.to_string().as_bytes().to_vec();
            let replace_utf8 = replace.as_bytes().to_vec();
            let split = split_vector_at(&string_utf8, find_utf8[0]);
            let split_vec: Vec<_> = split
                .iter()
                .map(|a| String::from_utf8(a.to_vec()).unwrap())
                .collect();
            let mut new_string_vec = vec![];
            if operation == "all" {
                for (n, _) in split_vec.iter().enumerate() {
                    if n > 0 {
                        let x = split_vec[n][1..].to_string();
                        new_string_vec.push(format!(
                            "{}{}",
                            String::from_utf8(replace_utf8.clone()).unwrap(),
                            x.clone()
                        ));
                    } else {
                        new_string_vec.push(split_vec[n].clone());
                    }
                }
            } else {
                if operation == "first" {
                    for (n, _) in split_vec.iter().enumerate() {
                        if n == 1 {
                            let x = split_vec[n][1..].to_string();

                            new_string_vec.push(format!(
                                "{}{}",
                                String::from_utf8(replace_utf8.clone()).unwrap(),
                                x.clone()
                            ));
                        } else {
                            new_string_vec.push(split_vec[n].clone());
                        }
                    }
                } else {
                    panic!("Either pass operation as `all` or `first`");
                }
            }
            new_string_vec.concat()
        } else {
            panic!("The character to replace does not exist in the string passed, please check!")
        }
    }
}

pub fn extract_vowels_consonants(string: String) -> (Vec<char>, Vec<char>) {
    /*
    Returns a tuple of vectors containing chars (after converting to lowercase)
    .0 : list of vowels
    .1 : list fo consonants
    */
    let bytes: Vec<_> = string.as_bytes().to_vec();
    let vowels: Vec<_> = bytes
        .iter()
        .filter(|a| {
            **a == 97
                || **a == 101
                || **a == 105
                || **a == 111
                || **a == 117
                || **a == 65
                || **a == 69
                || **a == 73
                || **a == 79
                || **a == 85
        })
        .collect();
    let consonants: Vec<_> = bytes
        .iter()
        .filter(|a| {
            **a != 97
                && **a != 101
                && **a != 105
                && **a != 111
                && **a != 117
                && **a != 65
                && **a != 69
                && **a != 73
                && **a != 79
                && **a != 85
                && ((**a > 96 && **a < 123) || (**a > 64 && **a < 91))
        })
        .collect();
    let output: (Vec<_>, Vec<_>) = (
        String::from_utf8(vowels.iter().map(|a| **a).collect())
            .unwrap()
            .chars()
            .collect(),
        String::from_utf8(consonants.iter().map(|a| **a).collect())
            .unwrap()
            .chars()
            .collect(),
    );
    output
}

pub fn sentence_case(string: String) -> String {
    /*
    "The quick brown dog jumps Over the lazy fox" => "The Quick Brown Dog Jumps Over The Lazy Fox"
    */
    let lower = string.to_lowercase();
    let split: Vec<_> = lower.split(' ').collect();
    let mut output = vec![];
    for i in split.iter() {
        let char_vec: Vec<_> = i.chars().collect();
        let mut b = [0; 2];
        char_vec[0].encode_utf8(&mut b);
        output.push(format!(
            "{}{}",
            &String::from_utf8(vec![b[0] - 32 as u8]).unwrap()[..],
            &i[1..]
        ));
    }
    output.join(" ")
}

pub fn remove_stop_words(string: String) -> String {
    /*
    "Rust is a multi-paradigm programming language focused on performance and safety, especially safe concurrency.[15][16] Rust is syntactically similar to C++,[17] but provides memory safety without using garbage collection.\nRust was originally designed by Graydon Hoare at Mozilla Research, with contributions from Dave Herman, Brendan Eich, and others.[18][19] The designers refined the language while writing the Servo layout or browser engine,[20] and the Rust compiler. The compiler is free and open-source software dual-licensed under the MIT License and Apache License 2.0."
                                                                                                    |
                                                                                                    V
    "Rust multi-paradigm programming language focused performance safety, especially safe concurrency.[15][16] Rust syntactically similar C++,[17] provides memory safety without using garbage collection.\nRust originally designed Graydon Hoare Mozilla Research, contributions Dave Herman, Brendan Eich, others.[18][19] designers refined language writing Servo layout browser engine,[20] Rust compiler. compiler free open-source software dual-licensed MIT License Apache License 2.0."
         */
    // https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip (14/06/2020)
    let mut split: Vec<_> = string.split(' ').collect();
    let stop_words = vec![
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "you're",
        "you've",
        "you'll",
        "you'd",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "she's",
        "her",
        "hers",
        "herself",
        "it",
        "it's",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "that'll",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "don't",
        "should",
        "should've",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "aren't",
        "couldn",
        "couldn't",
        "didn",
        "didn't",
        "doesn",
        "doesn't",
        "hadn",
        "hadn't",
        "hasn",
        "hasn't",
        "haven",
        "haven't",
        "isn",
        "isn't",
        "ma",
        "mightn",
        "mightn't",
        "mustn",
        "mustn't",
        "needn",
        "needn't",
        "shan",
        "shan't",
        "shouldn",
        "shouldn't",
        "wasn",
        "wasn't",
        "weren",
        "weren't",
        "won",
        "won't",
        "wouldn",
        "wouldn't",
        "I",
        "Me",
        "My",
        "Myself",
        "We",
        "Our",
        "Ours",
        "Ourselves",
        "You",
        "You're",
        "You've",
        "You'll",
        "You'd",
        "Your",
        "Yours",
        "Yourself",
        "Yourselves",
        "He",
        "Him",
        "His",
        "Himself",
        "She",
        "She's",
        "Her",
        "Hers",
        "Herself",
        "It",
        "It's",
        "Its",
        "Itself",
        "They",
        "Them",
        "Their",
        "Theirs",
        "Themselves",
        "What",
        "Which",
        "Who",
        "Whom",
        "This",
        "That",
        "That'll",
        "These",
        "Those",
        "Am",
        "Is",
        "Are",
        "Was",
        "Were",
        "Be",
        "Been",
        "Being",
        "Have",
        "Has",
        "Had",
        "Having",
        "Do",
        "Does",
        "Did",
        "Doing",
        "A",
        "An",
        "The",
        "And",
        "But",
        "If",
        "Or",
        "Because",
        "As",
        "Until",
        "While",
        "Of",
        "At",
        "By",
        "For",
        "With",
        "About",
        "Against",
        "Between",
        "Into",
        "Through",
        "During",
        "Before",
        "After",
        "Above",
        "Below",
        "To",
        "From",
        "Up",
        "Down",
        "In",
        "Out",
        "On",
        "Off",
        "Over",
        "Under",
        "Again",
        "Further",
        "Then",
        "Once",
        "Here",
        "There",
        "When",
        "Where",
        "Why",
        "How",
        "All",
        "Any",
        "Both",
        "Each",
        "Few",
        "More",
        "Most",
        "Other",
        "Some",
        "Such",
        "No",
        "Nor",
        "Not",
        "Only",
        "Own",
        "Same",
        "So",
        "Than",
        "Too",
        "Very",
        "S",
        "T",
        "Can",
        "Will",
        "Just",
        "Don",
        "Don't",
        "Should",
        "Should've",
        "Now",
        "D",
        "Ll",
        "M",
        "O",
        "Re",
        "Ve",
        "Y",
        "Ain",
        "Aren",
        "Aren't",
        "Couldn",
        "Couldn't",
        "Didn",
        "Didn't",
        "Doesn",
        "Doesn't",
        "Hadn",
        "Hadn't",
        "Hasn",
        "Hasn't",
        "Haven",
        "Haven't",
        "Isn",
        "Isn't",
        "Ma",
        "Mightn",
        "Mightn't",
        "Mustn",
        "Mustn't",
        "Needn",
        "Needn't",
        "Shan",
        "Shan't",
        "Shouldn",
        "Shouldn't",
        "Wasn",
        "Wasn't",
        "Weren",
        "Weren't",
        "Won",
        "Won't",
        "Wouldn",
        "Wouldn't",
    ];
    split.retain(|a| stop_words.contains(a) == false);
    split
        .iter()
        .map(|a| String::from(*a))
        .collect::<Vec<String>>()
        .join(" ")
}

pub fn tokenize<'a>(string: String, symbol: &Vec<&'a str>) -> Vec<String> {
    /*
    Tokeninzes by space and if required by any other symbol passed in as a vector
    To use only space and nothing else, pass empty vector
    */

    let output1: Vec<&str> = string.split(" ").collect();
    let mut output2 = output1
        .iter()
        .map(|a| a.to_string())
        .collect::<Vec<String>>();
    let mut output = vec![];
    for j in symbol.iter() {
        for i in output2.iter() {
            if i.contains(j) {
                output.push((*i).split(j).collect());
            } else {
                output.push(i.to_string());
            }
        }
        output2 = output.clone();
        output = vec![];
    }
    output2
}

/*
DESCRIPTION
-----------------------------------------
STRUCTS
-------

FUNCTIONS
---------
1. autocorrelation : at a given lag
    > 1. ts : &Vec<f64>
    > 2. lag : usize
    = Result<f64, std::io::Error>

2. simple_ma
    > 1. ts : &Vec<f64>
    > 2. lag : usize
    = Vec<f64>

3. exp_ma
    > 1. ts : &Vec<f64>
    > 2. alpha : f64
    = Vec<f64>

4. best_fit_line : returns intercept and slope of the best fit line
    > x: &Vec<f64>
    > y: &Vec<f64>)
     = (f64, f64)

*/

pub fn acf(ts: &Vec<f64>, lag: usize) -> Result<f64, std::io::Error> {
    /*
    To check for randomness of time series
    To check if the values are dependent on its past value (correlated)
    */
    // https://www.itl.nist.gov/div898/handbook/eda/section3/eda35c.htm
    let mean = mean(ts);
    let mut numerator = 0.;
    let mut denominator = 0.;
    for i in 0..ts.len() - lag {
        if i > lag {
            numerator += (ts[i] - mean) * (ts[i - lag] - mean);
            denominator += (ts[i] - mean) * (ts[i] - mean);
        }
    }
    match denominator {
        x if x != 0. => {
            if ((numerator / denominator).abs() > 0.5) && (lag != 0) {
                print!("At {:?} lag the series seems to be correlated\t", lag)
            }
            Ok(numerator / denominator)
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Denominator is 0!",
        )),
    }
}

pub fn simple_ma(ts: &Vec<f64>, lag: usize) -> Vec<f64> {
    let mut output = vec![];
    for i in 0..lag {
        if lag + i <= ts.len() {
            let sub_ts = ts[i..lag + i].to_vec();
            output.push(sub_ts.iter().fold(0., |a, b| a + b) / sub_ts.len() as f64);
        }
    }
    pad_with_zero(&mut output, lag, "pre")
}

pub fn exp_ma(ts: &Vec<f64>, alpha: f64) -> Vec<f64> {
    // https://www.youtube.com/watch?v=k_HN0wOKDd0
    // assuming first forecast is == first actual value
    // new forecast  = aplha*actual old value+(1-alpha)*forecasted old value
    let mut output = vec![ts[0]];
    for (n, i) in ts[1..].to_vec().iter().enumerate() {
        output.push(alpha * i + (1. - alpha) * output[n]);
    }
    // removing the last value and adding 0 in front
    let exp_ma = pad_with_zero(&mut output[..ts.len() - 1].to_vec(), 1, "pre");
    let mse = mean(
        &ts[1..]
            .to_vec()
            .iter()
            .zip(output[..ts.len() - 1].to_vec().iter())
            .map(|(a, b)| (a - b) * (a - b))
            .collect(),
    );
    println!("Mean square error of this forecasting : {:?}", mse);
    exp_ma
}

pub fn best_fit_line(x: &Vec<f64>, y: &Vec<f64>) -> (f64, f64) {
    // https://pythonprogramming.net/how-to-program-best-fit-line-machine-learning-tutorial/
    // intercept , slope
    let xy = x
        .iter()
        .zip(y.iter())
        .map(|a| a.0 * a.1)
        .collect::<Vec<f64>>();
    let xx = x
        .iter()
        .zip(x.iter())
        .map(|a| a.0 * a.1)
        .collect::<Vec<f64>>();
    let m = ((mean(x) * mean(y)) - mean(&xy)) / ((mean(x) * mean(x)) - mean(&xx));

    let b = mean(y) - m * mean(x);
    (b, m)
}
