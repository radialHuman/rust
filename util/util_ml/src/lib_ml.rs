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
        let (columns, values) = read_csv(self.file_path.clone()); // output is row wise

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
        confuse_me(&predicted, &y_test);
    }
    

    pub fn predict(test_features: &Vec<Vec<f64>>, weights: &Vec<f64>, threshold: f64) -> Vec<f64> {
        let length = test_features[0].len();
        let intercept = vec![vec![1.; length]];
        let new_x_test = [&intercept[..], &test_features[..]].concat();
        let mut pred = BLR::sigmoid(&new_x_test, weights);
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


pub struct KNN<'a>{
    pub file_path: String,
    pub test_size: f64,
    pub target_column: usize,
    pub k: usize,
    pub method: &'a str
}
impl<'a> KNN<'a>{
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
            let dis = Distance{row1:i.clone(),row2:j.clone()};
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

pub struct Distance{
    pub row1: Vec<f64>,
    pub row2: Vec<f64>,
}
impl Distance{
pub fn distance_euclidean(&self) -> f64 {
    // sqrt(sum((row1-row2)**2))

    let distance = self.row1
        .iter()
        .zip(self.row2.iter())
        .map(|(a, b)| (*a - *b) * (*a - *b))
        .collect::<Vec<f64>>();
    distance.iter().fold(0., |a, b| a + b).sqrt()
}

pub fn distance_manhattan(&self) -> f64 {
    // sum(|row1-row2|)

    let distance = self.row1
        .iter()
        .zip(self.row2.iter())
        .map(|(a, b)| (*a - *b).abs())
        .collect::<Vec<f64>>();
    distance.iter().fold(0., |a, b| a + b)
}

pub fn distance_cosine(&self) -> f64 {
    // 1- (a.b)/(|a||b|)

    let numerator = self.row1
        .iter()
        .zip(self.row2.iter())
        .map(|(a, b)| (*a * *b))
        .collect::<Vec<f64>>()
        .iter()
        .fold(0., |a, b| a + b);
    let denominator = (self.row1
        .iter()
        .map(|a| a * a)
        .collect::<Vec<f64>>()
        .iter()
        .fold(0., |a, b| a + b)
        .sqrt())
        * (self.row2
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
    let distance = self.row1
        .iter()
        .zip(self.row2.iter())
        .map(|(a, b)| (*a - *b).abs())
        .collect::<Vec<f64>>();
    distance.iter().cloned().fold(0. / 0., f64::max)
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
    println!("Number of rows = {}", rows - 1);
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

pub fn randomize_vector<T:std::clone::Clone>(rows: &Vec<T>) -> Vec<T> {
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

pub fn randomize<T:std::clone::Clone>(rows: &Vec<Vec<T>>) -> Vec<Vec<T>> {
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

pub fn how_many_and_where<T>(matrix: &Vec<Vec<T>>, number: T) -> Vec<(usize,usize)>
where
    T: std::cmp::PartialEq + std::fmt::Debug + Copy,
{
    /*
    Returns the positions of the number to be found in a matrix
    */
    let mut output = vec![];
    for (n,i) in matrix.iter().enumerate(){
        for j in how_many_and_where_vector(&i, number)
            {
            output.push((n,j));
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
    println!("{:?} : {:?}x{:?}", words, m.len(), m[0].len());
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
        matrix
            .iter()
            .map(|a| {
                a.iter()
                    .map(|b| (*b).replace("\r", "").parse::<f64>().unwrap())
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>()
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

            _ => println!(
                "Using the actual values without preprocessing unless 's' or 'm' is passed"
            ),
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

pub fn confuse_me(predicted: &Vec<f64>, actual: &Vec<f64>) {
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
            if **i == 0.0 && **j == 0.0 {
                tp += 1.;
            }
            if **i == 1.0 && **j == 1.0 {
                tng += 1.;
            }
            if **i == 0.0 && **j == 1.0 {
                fp += 1.;
            }
            if **i == 1.0 && **j == 0.0 {
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