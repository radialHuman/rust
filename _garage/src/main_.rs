fn main_() {

    // OUT OF ORDER FUNCTIONS
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
    pub fn confuse_me(&self, training_weights: &Vec<f64>) {
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
        let (train, test) = MultivariantLinearRegression::train_test_split(&df_f, self.split_ratio);
        println!("Train size: {}\nTest size : {:?}", train.len(), test.len());

        // feature and target split
        let mut train_feature = BTreeMap::new();
        let mut test_feature = BTreeMap::new();
        let mut train_target = BTreeMap::new();
        let mut test_target = BTreeMap::new();
        // let mut coefficients = vec![];

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

        let coefficients = vec![0.; train[0].len() - 1];
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
            MultivariantLinearRegression::generate_score(
                &predicted_values,
                &actual,
                self.header.len()
            )
            .0
        );
        println!(
            "The adjusted r2 of this model is : {:?}",
            MultivariantLinearRegression::generate_score(
                &predicted_values,
                &actual,
                self.header.len()
            )
            .1
        );
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
        // use rand::thread_rng;
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

    fn generate_score(predicted: &Vec<f64>, actual: &Vec<f64>, features: usize) -> (f64, f64) {
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
        let mut hypothesis_value;
        let mut cost_history = vec![];
        let mut loss;
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
