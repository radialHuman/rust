/*
DESCRIPTION
-----------------------------------------
STRUCTS
-------
1. MultivariantLinearRegression : header: Vec<String>, data: Vec<Vec<String>>, split_ratio: f64, alpha_learning_rate: f64, iterations: i32,
    > multivariant_linear_regression
    > batch_gradient_descent
    > hash_to_table
    x generate_score
    x mse_cost_function
    x train_test_split
    x randomize

2. BinaryLogisticRegression_f : features: Vec<Vec<f64>>,target: Vec<f64>,learning_rate: f64,iterations: u32,
    > train_test_split
    > read_n_split_n_shuffle
    x randomize
    x weightInitialization
    > sigmoid_activation
    x model_optimize
    > model_predict
    > pred_test
    > confusion_me

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

11. normalize_vector_f : between [0.,1.]
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

15. randomize_vector_f :
    1. > rows : &Vec<f64>
    = Vec<f64>

16. randomize_f :
    1. > rows : &Vec<Vec<f64>>
    = Vec<Vec<f64>>

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

*/

use crate::lib_matrix;
use lib_matrix::*;

pub struct MultivariantLinearRegression {
    pub header: Vec<String>,
    pub data: Vec<Vec<String>>,
    pub split_ratio: f64,
    pub alpha_learning_rate: f64,
    pub iterations: i32,
}

use std::collections::BTreeMap;
impl MultivariantLinearRegression {
    //
    // https://medium.com/we-are-orb/multivariate-linear-regression-in-python-without-scikit-learn-7091b1d45905
    pub fn multivariant_linear_regression(&self)
    //-> (Vec<f64>, Vec<f64>)
    {
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
        let mut train_feature = BTreeMap::new();
        let mut test_feature = BTreeMap::new();
        let mut train_target = BTreeMap::new();
        let mut test_target = BTreeMap::new();
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
        let mut norm_test_features = BTreeMap::new();
        let mut norm_train_features = BTreeMap::new();
        let mut norm_test_target = BTreeMap::new();
        let mut norm_train_target = BTreeMap::new();
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

        coefficients = vec![0.; train[0].len() - 1];
        let target: Vec<_> = norm_train_target.values().cloned().collect();
        // println!("TARGET\n{:?}", target[0].len());
        let (coefficeints, _) = MultivariantLinearRegression::batch_gradient_descent(
            &MultivariantLinearRegression::hash_to_table(&norm_train_features),
            &target[0],
            &coefficients,
            self.alpha_learning_rate,
            self.iterations,
        );
        println!("The weights of the inputs are {:?}", coefficeints);
        let pv: Vec<_> = MultivariantLinearRegression::hash_to_table(&norm_test_features)
            .iter()
            .map(|a| element_wise_operation(a, &coefficeints, "mul"))
            .collect();

        let mut predicted_values = vec![];
        for i in pv.iter() {
            predicted_values.push(i.iter().fold(0., |a, b| a + b))
        }

        let a = &MultivariantLinearRegression::hash_to_table(&norm_test_target);
        let mut actual = vec![];
        for i in a.iter() {
            actual.push(i[0]);
        }

        println!(
            "The r2 of this model is : {:?}",
            MultivariantLinearRegression::generate_score(&predicted_values, &actual)
        );
    }

    fn generate_score(predicted: &Vec<f64>, actual: &Vec<f64>) -> f64 {
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
        r2
    }

    fn mse_cost_function(features: &Vec<Vec<f64>>, target: &Vec<f64>, theta: &Vec<f64>) -> f64 {
        let rows = target.len();
        let prod = matrix_vector_product_f(&features, theta);
        // println!(">>>>>>>>\n{:?}x{:?}", prod.len(), target.len(),);
        let numerator: Vec<_> = element_wise_operation(&prod, target, "sub")
            .iter()
            .map(|a| *a * *a)
            .collect();
        // print!(".");
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
                "sub",
            )
            .clone();

            cost_history.push(MultivariantLinearRegression::mse_cost_function(
                features, target, &new_theta,
            ));
        }
        println!("");
        (new_theta.clone(), cost_history)
    }

    fn train_test_split(input: &Vec<Vec<f64>>, percentage: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // shuffle
        let data = MultivariantLinearRegression::randomize(input);
        // println!("{:?}", data);
        // split
        let test_count = (data.len() as f64 * percentage) as usize;
        // println!("Test size is {:?}", test_count);

        let test = data[0..test_count].to_vec();
        let train = data[test_count..].to_vec();
        (train, test)
    }

    fn randomize(rows: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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

    pub fn hash_to_table<T: Copy + std::fmt::Debug>(d: &BTreeMap<String, Vec<T>>) -> Vec<Vec<T>> {
        // changes the order of table columns
        let mut vector = vec![];
        for (_, v) in d.iter() {
            vector.push(v.clone());
        }
        let mut original = vec![];
        for i in 0..vector[0].len() {
            let mut row = vec![];
            for j in vector.iter() {
                row.push(j[i]);
            }
            original.push(row);
        }
        original
    }
}

#[derive(Debug)]
pub struct BinaryLogisticRegression_f {
    // https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc
    pub features: Vec<Vec<f64>>,
    pub target: Vec<f64>,
    pub learning_rate: f64,
    pub iterations: u32,
}

impl BinaryLogisticRegression_f {
    pub fn train_test_split(
        &self,
        test_percentage: f64,
    ) -> (BinaryLogisticRegression_f, BinaryLogisticRegression_f) {
        let training_rows =
            round_off_f(self.features[0].len() as f64 * (1. - test_percentage), 0) as usize;

        (
            // train
            BinaryLogisticRegression_f {
                features: self
                    .features
                    .iter()
                    .map(|a| a[0..training_rows - 1].to_vec())
                    .collect(),
                target: self.target[0..training_rows - 1].to_vec(),
                learning_rate: self.learning_rate,
                iterations: self.iterations,
            },
            // test
            BinaryLogisticRegression_f {
                features: self
                    .features
                    .iter()
                    .map(|a| a[training_rows..].to_vec())
                    .collect(),
                target: self.target[training_rows..].to_vec(),
                learning_rate: self.learning_rate,
                iterations: self.iterations,
            },
        )
    }

    pub fn read_n_split_n_shuffle(
        path: &str,
        target_header: &str,
    ) -> (Vec<Vec<String>>, Vec<Vec<String>>) {
        /*
        reads a txt or a csv and returns as a vector of vector of string
        the first one is training data
        split based on header passed as argument
        */
        let (headers, mut values) = read_csv(path.to_string()); // reading csv as vector of string
                                                                // shuffle data
        values = BinaryLogisticRegression_f::randomize(&values);
        let mut target_position = values[0].len() - 1; // as default last column is the target
                                                       // updating the target column position
        if headers.contains(&target_header.to_string()) {
            for (n, i) in headers.iter().enumerate() {
                if *i == target_header {
                    target_position = n;
                }
            }

            let mut data = vec![];
            let mut header = vec![];

            for j in 0..values[0].len() {
                if j != target_position {
                    let mut columns = vec![];
                    for i in values.iter() {
                        columns.push(i[j].clone());
                    }
                    data.push(columns);
                } else {
                    let mut columns = vec![];
                    for i in values.iter() {
                        columns.push(i[j].clone());
                    }
                    header.push(columns);
                }
            }
            (data, header)
        } else {
            panic!("Target not found in {:?}, please check", headers);
        }
    }

    fn randomize<T: std::clone::Clone>(rows: &Vec<Vec<T>>) -> Vec<Vec<T>> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
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
    fn weightInitialization(train: Vec<Vec<f64>>) -> (Vec<f64>, f64) {
        let w = vec![0.; train.len()];
        let b = 0.;
        (w, b)
    }

    pub fn sigmoid_activation(list: &Vec<f64>) -> Vec<f64> {
        list.iter().map(|a| 1. / ((a * -1.).exp() + 1.)).collect()
    }

    fn model_optimize(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        /*
        Returns
        derivative of weight
        derivative of bias
        cost function's result
        */
        // m = X.shape[0]
        let m = self.features[0].len() as f64;
        let (w, b) = BinaryLogisticRegression_f::weightInitialization(self.features.clone());

        // sigmoid_activation(np.dot(w,X.T)+b)
        let dot: Vec<_> = matrix_vector_product_f(&transpose(&self.features), &w)
            .iter()
            .map(|a| a + b)
            .collect();
        let final_result = BinaryLogisticRegression_f::sigmoid_activation(&dot);
        // cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))
        let cost = vector_addition(
            &mut element_wise_operation(
                &final_result
                    .iter()
                    .map(|a| a.log(1.0_f64.exp()))
                    .collect::<Vec<f64>>(),
                &self.target,
                "mul",
            ),
            &mut element_wise_operation(
                &final_result
                    .iter()
                    .map(|a| (1. - a).log(1.0_f64.exp()))
                    .collect::<Vec<f64>>(),
                &self.target.iter().map(|a| 1. - a).collect(),
                "mul",
            ),
        )
        .iter()
        .map(|a| a * (-1. / m))
        .collect::<Vec<f64>>();

        // dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T))
        let mut dw1 = vec![];
        for i in self.features.iter() {
            dw1.push(dot_product(
                &i,
                &final_result
                    .iter()
                    .zip(self.target.clone())
                    .map(|(a, b)| a - b)
                    .collect::<Vec<f64>>(),
            ));
        }
        let dw = dw1.iter().map(|a| (1. / m) * a).collect::<Vec<f64>>();
        // db = (1/m)*(np.sum(final_result-Y.T))
        let db = element_wise_operation(&self.target.clone(), &final_result, "sub")
            .iter()
            .map(|a| a * (-1. / m))
            .collect::<Vec<f64>>();
        (dw, db, cost)
    }

    pub fn model_predict(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
        let mut costs = vec![];
        let (mut w, mut b_value) =
            BinaryLogisticRegression_f::weightInitialization(self.features.clone());
        let mut b = vec![b_value];
        let mut dw = vec![];
        let mut db = vec![];
        let mut cost = vec![];

        print!("Calculating coefficients...");
        for i in 0..self.iterations {
            dw = BinaryLogisticRegression_f::model_optimize(self).0;
            db = BinaryLogisticRegression_f::model_optimize(self).1;
            cost = BinaryLogisticRegression_f::model_optimize(self).2;

            // w = w - (learning_rate * (dw.T))
            w = w
                .iter()
                .zip(dw.iter().map(|a| self.learning_rate * a))
                .collect::<Vec<(&f64, f64)>>()
                .iter()
                .map(|(a, b)| *a - b)
                .collect::<Vec<f64>>();
            // b = b - (learning_rate * db)
            b = db
                .iter()
                .map(|a| (self.learning_rate * a) - b[0])
                .collect::<Vec<f64>>();

            if i % 100 == 0 {
                costs.push(cost);
                print!("..");
            }
        }
        println!();
        (w, b, dw, db, costs)
    }

    pub fn pred_test(&self, training_weights: &Vec<f64>) -> Vec<f64> {
        // multiplying every column with corresponding coefficient
        let mut weighted_features: Vec<Vec<f64>> = vec![];
        for (n, j) in training_weights.iter().enumerate() {
            // let mut columns: Vec<Vec<f64>> = vec![];
            for (m, i) in self.features.iter().enumerate() {
                if n == m {
                    weighted_features.push(i.iter().map(|a| a * j).collect())
                }
            }
            // weighted_features.push(columns);
        }
        // println!("{:?}", weighted_features);
        // adding all the rows to get a probability
        let mut row_wise_addition = vec![];
        for n in 0..weighted_features[0].len() {
            let mut rows = vec![];
            for i in weighted_features.iter() {
                rows.push(i[n]);
            }
            row_wise_addition.push(rows.iter().fold(0., |a, b| a + b))
        }
        // turning probabilty into class
        BinaryLogisticRegression_f::sigmoid_activation(&row_wise_addition)
            .iter()
            .map(|a| round_off_f(*a, 0))
            .collect()
    }

    // pub fn find_accuracy(&self, training_weights: &Vec<f64>) {
    //     let prediction = self.pred_test(training_weights);
    //     let accuracy: Vec<_> = prediction
    //         .iter()
    //         .zip(self.target.clone())
    //         // .collect::<Vec<(&f64, f64)>>()
    //         .map(|(a, b)| if *a == b { 1 } else { 0 })
    //         .collect::<Vec<i32>>();

    //     let correct_prediction = accuracy.iter().fold(0, |a, b| a + b) as f64;
    //     let length = self.target.len() as f64;
    //     let output = correct_prediction / length;
    //     // println!("Accuracy {:.3}", output);
    // }
    pub fn confusion_me(&self, training_weights: &Vec<f64>) {
        // https://medium.com/@MohammedS/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b
        let prediction = self.pred_test(training_weights);
        let mut tp = 0.; // class_one_is_class_one
        let mut fp = 0.; // class_one_is_class_two(Type 1)
        let mut fng= 0.; // class_two_is_class_one (Type 1)
        let mut tng = 0.; // class_two_is_class_two

        for (i, j) in self
            .target
            .iter()
            .zip(prediction.iter())
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
                fng+= 1.;
            }
        }
        println!("|------------------------|");
        println!(
            "|  {:?}    |   {:?}",
            tp, fp
        );
        println!("|------------------------|");
        println!(
            "|  {:?}    |   {:?}",
            fng, tng
        );
        println!("|------------------------|");
        println!("Accuracy : {:.3}",(tp + tng)/(tp + fp + fng + tng) );
        println!("Precision : {:.3}",(tp)/(tp + fp));
        let precision:f64 = (tp)/(tp + fp);
        println!("Recall (sensitivity) : {:.3}",(tp)/(tp + fng));
        let recall:f64 = (tp)/(tp + fng);
        println!("Specificity: {:.3}",(tng)/(fp + tng));
        println!("F1 : {:.3}\n\n",(2.*precision*recall)/(precision*recall));
        

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
    let mu1 = mean(list1);
    let mu2 = mean(list2);
    let zero: T = "0".parse().unwrap();
    let _len_str: f64 = list1.len().to_string().parse().unwrap(); // if division is required
    let tupled: Vec<_> = list1.iter().zip(list2).collect();
    let output = tupled.iter().fold(zero, |a, b| {
        a + ((*b.0 - mu1.to_string().parse().unwrap()) * (*b.1 - mu2.to_string().parse().unwrap()))
    });
    let numerator: f64 = output.to_string().parse().unwrap();
    numerator / _len_str
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
    let train_features = &train.iter().map(|a| a.0).collect();
    let test_features = &test.iter().map(|a| a.1).collect();
    let (offset, slope) = coefficient(train_features, test_features);
    let b0: T = offset.to_string().parse().unwrap();
    let b1: T = slope.to_string().parse().unwrap();
    let predicted_output = test.iter().map(|a| b0 + b1 * a.0).collect();
    let original_output: Vec<_> = test.iter().map(|a| a.0).collect();
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
    println!("========================================================================================================================================================");
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
    println!("========================================================================================================================================================");
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
    println!("========================================================================================================================================================");
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

pub fn normalize_vector_f(list: &Vec<f64>) -> Vec<f64> {
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

    let first_calc_T = transpose(&first_calc);
    let mut X = vec![];
    for j in 0..matrix1[0].len() {
        let mut row = vec![];
        for i in matrix1.iter() {
            row.push(i[j]);
        }
        X.push(row);
    }

    // PYTHON : // final_calc = np.dot(first_calc.T, X)
    let mut final_calc = vec![];
    for i in first_calc_T.iter() {
        for j in X.iter() {
            final_calc.push(dot_product(&i, &j))
        }
    }

    // println!("{:?}\n{:?}", &first_calc_T, &X);
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

pub fn randomize_vector_f(rows: &Vec<f64>) -> Vec<f64> {
    /*
    Shuffle values inside vector
    */
    use rand::seq::SliceRandom;
    use rand::{thread_rng, Rng};
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

pub fn randomize_f(rows: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    /*
    Shuffle rows inside matrix
    */
    use rand::seq::SliceRandom;
    use rand::{thread_rng, Rng};
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
        use rand::seq::SliceRandom;
        use rand::{thread_rng, Rng};
        let mut order: Vec<usize> = (0..rows.len()  as usize).collect();
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
    let data = randomize_vector_f(input);
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
    let data = randomize_f(input);
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
    let cov = covariance(list1, list2);
    let output = match name {
        "p" => cov / (std_dev(list1) * std_dev(list2)),
        "s" => {
            // https://statistics.laerd.com/statistical-guides/spearmans-rank-order-correlation-statistical-guide-2.php
            //covariance(&rank(list1), &rank(list2))/(std_dev(&rank(list1))*std_dev(&rank(list2)))
            let ranked_list1 = s_rank(list1);
            let ranked_list2 = s_rank(list2);
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

pub fn s_rank<T>(list1: &Vec<T>) -> Vec<(T, f64)>
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
    let mut rank: Vec<_> = repeats
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
    Returns the positions of the number to be found in a matrix
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
