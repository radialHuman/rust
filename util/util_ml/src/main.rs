mod lib;
use lib::*;

fn main() {
    println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    println!("                                                              LIB_NN");
    println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    let layer = LayerDetails {
        n_inputs: 10,
        n_neurons: 5,
    };
    println!(
        "Bias of {:?} is introduced to all the neurons => {:?}\n\n",
        1.,
        layer.create_bias(1.)
    );

    print_a_matrix(
        "Random weights introduced to all the neurons and input [10x5] =>",
        &layer.create_weights(),
    );

    print_a_matrix(
        "(To check randomness) Random weights introduced to all the neurons and input [10x5] =>",
        &layer.create_weights(),
    );

    let input = vec![
        vec![23., 45., 12., 45.6, 218., -12.7, -19., 2., 5.8, 2.],
        vec![3., 5., 12.5, 456., 28.1, -12.9, -19.2, 2.5, 8., 222.],
        vec![13., 4., 12.7, 5.6, 128., -12.1, -19.2, 15.2, 54., 32.],
        vec![73., 45.4, 120., 4.6, 8., -1., -19.2, 23.8, 10., 22.],
        vec![27., 4.5, 1., 4.6, 8., -1., -19.2, 2.7, 2.5, 12.],
    ];
    print_a_matrix("For input\n", &input);
    print_a_matrix("Weights\n", &layer.create_weights());
    println!("Bias\n{:?}\n\n", &layer.create_bias(0.));
    println!("Using TanH");
    print_a_matrix(
        "The output of the layer is \n",
        &layer.output_of_layer(
            &input,
            &layer.create_weights(),
            &mut layer.create_bias(1.),
            "tanh",
            0.,
        ),
    );
    println!("\n\n");
    println!("Using Sigmoid");
    print_a_matrix(
        "The output of the layer is \n",
        &layer.output_of_layer(
            &input,
            &layer.create_weights(),
            &mut layer.create_bias(1.),
            "sigmoid",
            0.,
        ),
    );
    println!("\n\n");
    println!("Using ReLU");
    print_a_matrix(
        "The output of the layer is \n",
        &layer.output_of_layer(
            &input,
            &layer.create_weights(),
            &mut layer.create_bias(1.),
            "relu",
            0.,
        ),
    );
    println!("\n\n");
    println!("Using Leaky ReLU");
    print_a_matrix(
        "The output of the layer is \n",
        &layer.output_of_layer(
            &input,
            &layer.create_weights(),
            &mut layer.create_bias(1.),
            "leaky relu",
            0.1,
        ),
    );
    println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    println!("                                                              LIB_STRING");
    println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

    let stm = StringToMatch {
        string1: String::from("42, Some street, Area51, Ambience"),
        string2: String::from("42 Street, Area, Ambiance"),
    };
    println!();
    println!(
        "Comparing {:?} and {:?} with 1:2 weightage for position:presence => {:?}",
        stm.string1,
        stm.string2,
        stm.compare_percentage(1., 2.)
    );
    println!();
    println!(
        "Comparing {:?} and {:?} with 2:1 weightage for position:presence => {:?}",
        stm.string1,
        stm.string2,
        stm.compare_percentage(2., 1.)
    );
    println!();
    println!(
        "{:?} after cleaning becomes => {:?}",
        stm.string1,
        StringToMatch::clean_string(stm.string1.clone())
    );
    println!();
    println!(
        "Character count of {:?} =>\n{:?}",
        stm.string1,
        StringToMatch::char_count(stm.string1.clone())
    );
    println!();
    println!(
        "Most frequent character in {:?} => {:?}",
        stm.string1,
        StringToMatch::frequent_char(stm.string1.clone())
    );
    println!();
    println!(
        "Most frequent character replaced with `XXX` in  {:?} => {:?}",
        stm.string1,
        StringToMatch::char_replace(
            stm.string1.clone(),
            StringToMatch::frequent_char(stm.string1.clone()),
            "XXX".to_string(),
            "all"
        )
    );
    println!();
    println!(
        "Splitting numebrs and aplhabets in {:?} => {:?}",
        stm.string1,
        StringToMatch::split_alpha_numericals(stm.string1.clone())
    );
    println!();
    println!(
        "Fuzzy matched percentage (n-gram = 2) of {:?} and {:?} => {:?}",
        stm.string1,
        stm.string2,
        stm.fuzzy_subset(2)
    );
    println!();
    println!(
        "Fuzzy matched percentage (n-gram = 3) of {:?} and {:?} => {:?}",
        stm.string1,
        stm.string2,
        stm.fuzzy_subset(3)
    );

    let string = String::from("The quick brown dog jumps Over the lazy fox");
    println!(
        "{:?}\nhas these vowels\n{:?}\nand these consonants\n{:?}\n\n",
        string,
        extract_vowels_consonants(string.clone()).0,
        extract_vowels_consonants(string.clone()).1
    );
    println!(
        "Sentence case of {:?} is\n {:?}\n\n",
        string.clone(),
        sentence_case(string.clone())
    );
    let string2 = String::from("Rust is a multi-paradigm programming language focused on performance and safety, especially safe concurrency.[15][16] Rust is syntactically similar to C++,[17] but provides memory safety without using garbage collection.
Rust was originally designed by Graydon Hoare at Mozilla Research, with contributions from Dave Herman, Brendan Eich, and others.[18][19] The designers refined the language while writing the Servo layout or browser engine,[20] and the Rust compiler. The compiler is free and open-source software dual-licensed under the MIT License and Apache License 2.0.");

    println!(
        "Removing stop words from\n{:?}\ngives\n{:?}\n\n",
        string2.clone(),
        remove_stop_words(string2.clone())
    );

    println!();
    println!();
    println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    println!("                                                              LIB_MATRIX");
    println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    println!();
    println!();
    let mat = MatrixF {
        matrix: vec![
            vec![23., 45., 12., 45.6, 218.],
            vec![3., 5., 12.5, 456., 28.1],
            vec![13., 4., 12.7, 5.6, 128.],
            vec![73., 45.4, 120., 4.6, 8.],
            vec![27., 4.5, 1., 4.6, 8.],
        ],
    };
    print_a_matrix("This matrix will be used from here on", &mat.matrix);
    println!("The determinant is : {:?}\n", mat.determinant_f());
    println!(
        "Thats too much, the round off is : {:?}\n",
        round_off_f(mat.determinant_f(), 2)
    );
    println!(
        "Is this a square matrix? {:?}\n",
        MatrixF::is_square_matrix(&mat.matrix)
    );
    print_a_matrix("The Inverse is : ", &mat.inverse_f());
    print_a_matrix("The Transpose is : ", &transpose(&mat.matrix));

    let m1 = vec![vec![3; 5]; 5];
    print_a_matrix("This", &m1);
    print_a_matrix("Converted into floats", &make_matrix_float(&m1));
    print_a_matrix("Into", &mat.matrix);
    print_a_matrix(
        "is\n",
        &matrix_multiplication(&mat.matrix, &make_matrix_float(&m1)),
    );

    print_a_matrix(
        &format!("{:?}\nelement wise into\n{:?} is ", &mat.matrix, &m1),
        &element_wise_matrix_operation(&mat.matrix, &make_matrix_float(&m1), "mul"),
    );

    print_a_matrix(
        &format!("{:?}\nelement wise added\n{:?} is ", &mat.matrix, &m1),
        &element_wise_matrix_operation(&mat.matrix, &make_matrix_float(&m1), "add"),
    );

    print_a_matrix(
        &format!("{:?}\nelement wise subtracted\n{:?} is ", &mat.matrix, &m1),
        &element_wise_matrix_operation(&mat.matrix, &make_matrix_float(&m1), "sub"),
    );

    print_a_matrix(
        &format!("{:?}\nelement wise divided\n{:?} is ", &mat.matrix, &m1),
        &element_wise_matrix_operation(&mat.matrix, &make_matrix_float(&m1), "div"),
    );

    println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>  Vectors");
    let mut v: Vec<_> = "abcdefghijkl".chars().collect();
    println!("This :{:?}", v);
    println!(
        "becomes with a few zeros :{:?}",
        &pad_with_zero(&mut v, 5, "post")
    );
    println!("or can also become :{:?}", &pad_with_zero(&mut v, 5, "pre"));
    print_a_matrix("becomes a 4x3 matrix", &shape_changer(&v, 3, 4));

    let vi = vec![3; 10];
    println!("This {:?} becomes {:?}", &vi, &make_vector_float(&vi));
    let mut v1 = make_vector_float(&vi);
    let mut v2 = vec![2.; 10];
    let added_vector = vector_addition(&mut v1, &mut v2);
    let dotted = dot_product(&mut v1, &mut v2);
    let element_wise_multiplication = element_wise_operation(&mut v1, &mut v2, "mul");
    let element_wise_subtraction = element_wise_operation(&mut v1, &mut v2, "sub");
    let element_wise_division = element_wise_operation(&mut v1, &mut v2, "div");
    println!("{:?} + {:?} = {:?}\n", &mut v1, &mut v2, added_vector);
    println!("{:?} . {:?} = {:?}\n", &mut v1, &mut v2, dotted);
    println!(
        "{:?} * {:?} = {:?}\n",
        &mut v1, &mut v2, element_wise_multiplication
    );
    println!(
        "{:?} - {:?} = {:?}\n",
        &mut v1, &mut v2, element_wise_subtraction
    );
    println!(
        "{:?} / {:?} = {:?}\n",
        &mut v1, &mut v2, element_wise_division
    );

    println!(
        "Min and Max number in {:?} is {:?}]n",
        vec![23.0, 45.0, 12.0, 45.6, 218.0],
        min_max_f(&vec![23.0, 45.0, 12.0, 45.6, 218.0])
    );

    println!(
        "The numbers and their frequency in {:?} is\n{:?}\n",
        &pad_with_zero(&mut v, 5, "post"),
        &value_counts(&pad_with_zero(&mut v, 5, "post"))
    );

    println!(
        "Distinct numbers in  {:?} is\n{:?}\n",
        &pad_with_zero(&mut v, 5, "post"),
        &unique_values(&pad_with_zero(&mut v, 5, "post"))
    );

    println!(
        "{:?}\ninto\n{:?}\nis\n{:?}",
        &print_a_matrix("", &mat.matrix),
        &vec![3.; 5],
        &matrix_vector_product_f(&mat.matrix, &vec![3.; 5])
    );

    println!("\n\n{:?}", vec![3; 10]);
    print_a_matrix("breaks in to", &split_vector(&vec![3; 10], 5));
    print_a_matrix("or breaks in to", &split_vector(&vec![3; 10], 2));

    println!("\n{:?}", &v);
    print_a_matrix("splits at i", &split_vector_at(&v, 'i'));
    println!("\n{:?}", &vec![3; 10]);
    print_a_matrix("splits at 3", &split_vector_at(&vec![3; 10], 3));

    println!("0 is a {:?}", type_of(0));
    println!("0.0 is a {:?}", type_of(0.));
    println!("\"0\" is a {:?}", type_of("0"));
    println!("'0' is a {:?}\n\n", type_of('0'));

    let a = vec![vec![1, 2], vec![3, 5]];
    let b = vec![vec![0, 1], vec![5, 7]];
    println!("A: {:?}\nJOINED with\nB: {:?}",a,b);
    print_a_matrix("Wide", &join_matrix(&a, &b, "wide"));
    print_a_matrix("Long", &join_matrix(&a, &b, "long"));
    print_a_matrix("First 2 rows of long is ",  &head(&join_matrix(&a, &b, "long"),2));
    print_a_matrix("Last 2 rows of long is ",  &tail(&join_matrix(&a, &b, "long"),2));

    println!("Row to column and vice versa conversion");
    let c = vec![vec![1,6,11],vec![2,7,12],vec![3,8,13],vec![4,9,14],vec![5,10,15]];
    print_a_matrix("This\n",&c);
    print_a_matrix("becomes\n", &row_to_columns_conversion(&c));
    print_a_matrix("Which was originally\n", &columns_to_rows_conversion(&row_to_columns_conversion(&c)));

    let d = vec![vec!["A".to_string(),"B".to_string(),"C".to_string()],vec!["D".to_string(),"E".to_string(),"F".to_string()]];
    print_a_matrix("String matrix to &str matrix\n", &make_matrix_string_literal(&d));



    println!();
    println!();
    println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    println!("                                                              LIB_ML");
    println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    println!();
    println!();

    let vector1 = vec![23.0, 45.0, 12.0, 45.6, 218.0];
    let vector2 = vec![21.0, 4.0, 12.5, 32.6, 118.0];
    println!("The mean of {:?}  is {:?}", vector1, mean(&vector1));
    println!("The variance of {:?}  is {:?}", vector1, variance(&vector1));
    println!(
        "The covariance of {:?} and {:?} is {:?}",
        vector1,
        vector2,
        covariance(&vector1, &vector2)
    );
    println!(
        "The equation of {:?} and {:?} is b0 = {:?}, b1 = {:?}",
        vector1,
        vector2,
        coefficient(&vector1, &vector2).0,
        coefficient(&vector1, &vector2).1
    );

    println!("\n>>>>>>>>>>>>>>>>> Simple Linear Regression" );
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
    let predicted_output = simple_linear_regression_prediction(&to_train_on, &to_test_on);
    let original_output: Vec<_> = to_test_on.iter().map(|a| a.0).collect();
    println!(
        "Predicted is {:?}\nOriginal is {:?}\n",
        &predicted_output, &original_output
    );

    let df = read_csv("./data/dataset_iris.txt".to_string());
    println!("IRIS DATA:\n{:?}\n{:?}\n\n", df.0, df.1);

    let string_numbers = vec![1, 6, 2, 6, 8, 2, 23, 3, 5, 2, 4, 2, 0]
        .iter()
        .map(|a| String::from(a.to_string()))
        .collect();
    let conversion = convert_and_impute(&string_numbers, 0., 999.);
    let floating_petal_length = conversion.0.unwrap();
    let missing_value = conversion.1;
    println!(
        "{:?}\nis now\n{:?}\nwith missing values at\n{:?}",
        &string_numbers, floating_petal_length, missing_value
    );

    println!(
        "Categorized {:?} is now {:?}",
        &string_numbers,
        convert_string_categorical(&string_numbers.iter().map(|a| &*a).collect(), false)
    );

    let arr = vec![1., 2., 3., -5., -7., 0.];
    println!(
        "{:?}\nNormalized to : {:?}\n",
        arr,
        min_max_scaler(&arr)
    );
    let (min, max) = min_max_f(&arr);
    println!("In {:?}\nminimum is {} and maximum is {}\n", arr, min, max);

    let arranged = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
    println!(
        "Shuffling {:?} gives {:?}",
        arranged,
        randomize_vector(&arranged)
    );
    println!(
        "Shuffling again {:?} gives {:?}",
        arranged,
        randomize_vector(&arranged)
    );
    println!(
        "Shuffling again {:?} gives {:?}\n",
        arranged,
        randomize_vector(&arranged)
    );

    let arranged = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
    println!(
        "Shuffling {:?} gives {:?}",
        arranged,
        randomize_vector(&arranged)
    );
    println!(
        "Shuffling again {:?} gives {:?}",
        arranged,
        randomize_vector(&arranged)
    );
    println!(
        "Shuffling again {:?} gives {:?}\n",
        arranged,
        randomize_vector(&arranged)
    );

    println!(
        "Train and test vectors of {:?} with 50% test split is {:?}",
        arranged,
        train_test_split_vector_f(&arranged, 0.5)
    );
    println!(
        "Train and test vectors of {:?} with 20% test split is {:?}\n",
        arranged,
        train_test_split_vector_f(&arranged, 0.2)
    );

    let arranged_matrix = vec![
        vec![1., 2., 3., 4., 5.],
        vec![3., 5., 8., 1., 0.3],
        vec![0.5, 0.6, 0.1, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0],
        vec![1., 1., 1., 1., 1.],
    ];
    print_a_matrix("The original matrix", &arranged_matrix);
    print_a_matrix(
        &format!("{}", String::from("Shuffling gives")),
        &randomize(&arranged_matrix),
    );
    print_a_matrix(
        &format!("{}", String::from("Shuffling again gives")),
        &randomize(&arranged_matrix),
    );
    print_a_matrix(
        &format!("{}", String::from("Shuffling again gives")),
        &randomize(&arranged_matrix),
    );

    print_a_matrix(
        &format!("{}", String::from("Train matrix with 50% test split is")),
        &train_test_split_f(&arranged_matrix, 0.5).0,
    );
    print_a_matrix(
        &format!("{}", String::from("Train matrix with 20% test split is")),
        &train_test_split_f(&arranged_matrix, 0.2).0,
    );

    let v1 = vec![56, 75, 45, 71, 62, 64, 58, 80, 76, 61]
        .iter()
        .map(|a| *a as f64)
        .collect();
    let v2 = vec![66, 70, 40, 60, 65, 56, 59, 77, 67, 63]
        .iter()
        .map(|a| (*a * -1) as f64)
        .collect();
    println!(
        "Pearson Correlation of {:?} and {:?} : {:?}\n",
        &v1,
        &v2,
        correlation(&v1, &v2, "p")
    );
    println!("Spearman's rank of {:?} : {:?}", &v1, spearman_rank(&v1));
    println!(
        "Spearman's Correlation of {:?} and {:?} : {:?}\n",
        &v1,
        &v2,
        correlation(&v1, &v2, "s")
    );

    println!("Std. Dev of {:?} is {:?}\n", v1, std_dev(&v1));

    let arr = vec![1, 1, 1, 1, 4, 2, 2, 5, 2, 7, 5, 3, 3, 6, 7, 5, 4, 0, 0, 10];
    println!(
        "Position of {:?} in {:?} is {:?}\n",
        2,
        arr,
        how_many_and_where_vector(&arr, 2)
    );
    let list = vec![
        vec![
            0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4,
            0.3, 0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2,
            0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 1.4,
            1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0, 1.3, 1.4, 1.0, 1.5, 1.0, 1.4, 1.3, 1.4, 1.5, 1.0,
            1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7, 1.5, 1.0,
        ],
        vec![
            1.1, 1.0, 0.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3, 1.2, 1.4, 1.2, 1.0, 1.3, 1.2, 1.3,
            1.3, 1.1, 1.3, 2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8, 1.8, 2.5, 2.0, 1.9, 2.1, 2.0,
            2.4, 2.3, 1.8, 2.2, 2.3, 1.5, 2.3, 2.0, 2.0, 1.8, 2.1, 1.8, 1.8, 1.8, 2.1, 1.6, 1.9,
            2.0, 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9, 2.3, 2.5, 2.3, 1.9, 2.0,
            2.3, 1.8,
        ],
    ];
    print_a_matrix("In", &list);
    println!("0.2 can be found in {:?}\n",how_many_and_where(&list, 0.2));

    let sample = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    println!(
        "Z-score of {:?} in {:?} is {:?}\n",
        4,
        sample,
        z_score(&sample, 4)
    );

    let values = vec!["A","B","C"];
    println!("{:?}", values);
    print_a_matrix("One Hot Encoded", &one_hot_encoding(&values));

    // println!("\n>>>>>>>>>>>>>>>>> ORDINARY LEAST SQUARE" );
    let mut file = "./data/ccpp.csv".to_string();
    // let lr = OLS{
    //     file_path: file.clone(),
    //     target : 5,
    //     test_size: 0.20,
    // };
    // lr.fit();
    // println!();


    println!(">>>>>>>>>>>>>>>>> Simple Logistic Regression" );
    file = "./data/data_banknote_authentication.txt".to_string();
    let logistic = BLR{
        file_path: file.clone(), 
        test_size: 0.20, 
        target_column:5, 
        learning_rate:0.1, 
        iter_count:1000, 
        binary_threshold:0.5
        };
    logistic.fit();

    println!(">>>>>>>>>>>>>>>>> K-Nearest Neighbour" );
    file = "./data/data_banknote_authentication.txt".to_string();
    let knn = KNN{
        file_path: file.clone(),
        test_size: 0.20,
        target_column: 5,
        k: 10,
        method: "e"
        };
    knn.fit();
}

/*
OUTPUT

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                                              LIB_NN
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Bias of 1.0 is introduced to all the neurons => [1.0, 1.0, 1.0, 1.0, 1.0]


Random weights introduced to all the neurons and input [10x5] =>
[-0.518, 0.717, -0.623, 0.947, 0.078]
[0.031, 0.045, 0.823, -0.294, -0.445]
[0.616, -0.9, 0.347, -0.713, -0.102]
[-0.864, -0.197, 0.198, -0.116, -0.646]
[0.933, 0.922, 0.584, -0.316, 0.589]
[0.004, -0.425, 0.645, 0.738, 0.029]
[0.805, -0.893, 0.144, 0.401, 0.346]
[0.916, 0.105, -0.4, 0.128, 0.306]
[0.991, -0.406, -0.194, -0.676, 0.17]
[0.752, -0.049, 0.206, 0.388, -0.835]


(To check randomness) Random weights introduced to all the neurons and input [10x5] =>
[-0.34, 0.315, -0.502, -0.961, -0.453]
[0.42, -0.685, 0.22, 0.715, 0.953]
[0.045, -0.119, -0.507, -0.925, -0.888]
[-0.219, -0.648, 0.808, 0.774, -0.787]
[-0.072, -0.432, -0.135, -0.72, 0.241]
[0.387, 0.494, 0.172, -0.015, 0.23]
[-0.026, 0.332, 0.672, 0.972, -0.864]
[0.116, -0.418, 0.832, -0.82, 0.183]
[0.6, 0.211, -0.006, 0.664, 0.208]
[0.716, 0.692, 0.675, 0.924, 0.483]


For input

[23.0, 45.0, 12.0, 45.6, 218.0, -12.7, -19.0, 2.0, 5.8, 2.0]
[3.0, 5.0, 12.5, 456.0, 28.1, -12.9, -19.2, 2.5, 8.0, 222.0]
[13.0, 4.0, 12.7, 5.6, 128.0, -12.1, -19.2, 15.2, 54.0, 32.0]
[73.0, 45.4, 120.0, 4.6, 8.0, -1.0, -19.2, 23.8, 10.0, 22.0]
[27.0, 4.5, 1.0, 4.6, 8.0, -1.0, -19.2, 2.7, 2.5, 12.0]


Weights

[-0.436, -0.245, 0.889, 0.299, 0.976]
[-0.302, 0.312, 0.266, 0.863, 0.637]
[-0.401, 0.881, 0.525, -0.503, -0.318]
[-0.749, -0.124, 0.067, -0.298, 0.708]
[0.114, 0.22, -0.049, 0.666, -0.629]
[0.34, 0.867, 0.307, -0.524, 0.844]
[-0.741, -0.328, -0.991, 0.507, -0.342]
[-0.978, 0.07, -0.926, 0.941, 0.058]
[-0.887, 0.413, 0.976, -0.444, -0.275]
[-0.022, -0.477, -0.566, -0.576, 0.879]


Bias
[0.0, 0.0, 0.0, 0.0, 0.0]


Using TanH
Multiplication of 5x10 and 10x5
Output will be 5x5
Alpha is for 'leaky relu' only, it is not taken into account here
The output of the layer is

[1.0, 1.0, -0.9999999974044121, -1.0, -0.9992088701644279]
[-1.0, -1.0, -1.0, -1.0, -1.0]
[1.0, -1.0, 1.0, 1.0, 1.0]
[-1.0, 1.0, -1.0, -0.9999999999658566, 1.0]
[-1.0, 1.0, -1.0, -1.0, -0.9999999999999983]





Using Sigmoid
Multiplication of 5x10 and 10x5
Output will be 5x5
Alpha is for 'leaky relu' only, it is not taken into account here
The output of the layer is

[0.0000000000000000000000000000000000000000000018241774712110004, 0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010403835196584106, 0.000000000000000000000000000000000000007668481577670815, 1.0, 0.9999999516741899]
[0.9604182228869498, 0.0000000000000000000000000000000000000000000000000000000000000010363043332113411, 0.0000000000027169803982242745, 0.000000000000007864637104691795, 0.0000000000029215699208428217]
[0.00000000000000000000000000000000000000000000037992902680634475, 1.0, 0.00000000000000000000000000000000002382804737056865, 0.0000000000000000000000000000000000000008700702067007188, 0.0000014945259457844294]
[0.0000000000000000000000000000000000000000000000000000000266996458336415, 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000005240339548870849, 0.000000000000000000000000000000000005612315322139089, 0.000002774048440064603, 0.000000003524596048127927]
[1.0, 1.0, 1.0, 0.9999999999990392, 0.00038219398388281573]





Using ReLU
Multiplication of 5x10 and 10x5
Output will be 5x5
Alpha is for 'leaky relu' only, it is not taken into account here
The output of the layer is

[119.4902, 151.7663, 69.7772, 135.9422, 8.772200000000003]
[0.0, 109.04159999999999, 0.0, 0.0, 0.0]
[190.2846, 11.003699999999995, 116.3366, 46.647000000000006, 18.783399999999993]
[18.876100000000005, 0.0, 0.0, 0.0, 0.0]
[0.0, 0.0, 0.0, 0.0, 0.0]





Using Leaky ReLU
Multiplication of 5x10 and 10x5
Output will be 5x5
The output of the layer is

[126.9695, 6.048900000000003, 104.0939, -0.7891199999999995, 5.942500000000001]
[131.5997, 89.22589999999998, 3.5644999999999953, 115.177, -1.73442]
[50.20030000000001, 398.9471000000001, 30.405700000000003, 11.261000000000006, 37.139199999999995]
[-8.69261, -31.756259999999997, -6.11214, -1.915439999999999, -2.76422]
[-17.07507, -26.010680000000004, -5.44941, 26.1764, -0.2747300000000001]


>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                                              LIB_STRING
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Comparing "42, Some street, Area51, Ambience" and "42 Street, Area, Ambiance" with 1:2 weightage for position:presence => 101.92307692307692

Comparing "42, Some street, Area51, Ambience" and "42 Street, Area, Ambiance" with 2:1 weightage for position:presence => 71.15384615384615

"42, Some street, Area51, Ambience" after cleaning becomes => "42somestreetarea51ambience"

Character count of "42, Some street, Area51, Ambience" =>
{' ': 4, ',': 3, '1': 1, '2': 1, '4': 1, '5': 1, 'a': 3, 'b': 1, 'c': 1, 'e': 6, 'i': 1, 'm': 2, 'n': 1, 'o': 1, 'r': 2, 's': 2, 't': 2}

Most frequent character in "42, Some street, Area51, Ambience" => 'e'

Most frequent character replaced with `XXX` in  "42, Some street, Area51, Ambience" => "42, SomXXX strXXXXXXt, ArXXXa51, AmbiXXXncXXX"

[52, 50, 44, 32, 83, 111, 109, 101, 32, 115, 116, 114, 101, 101, 116, 44, 32, 65, 114, 101, 97, 53, 49, 44, 32, 65, 109, 98, 105, 101, 110, 99, 101]
Splitting numebrs and aplhabets in "42, Some street, Area51, Ambience" => ("4251", " Some street Area Ambience")

"42streetareaambiance" in "42somestreetarea51ambience"
Fuzzy matched percentage (n-gram = 2) of "42, Some street, Area51, Ambience" and "42 Street, Area, Ambiance" => 81.25

"42streetareaambiance" in "42somestreetarea51ambience"
Fuzzy matched percentage (n-gram = 3) of "42, Some street, Area51, Ambience" and "42 Street, Area, Ambiance" => 68.75
"The quick brown dog jumps Over the lazy fox"
has these vowels
['e', 'u', 'i', 'o', 'o', 'u', 'O', 'e', 'e', 'a', 'o']
and these consonants
['T', 'h', 'q', 'c', 'k', 'b', 'r', 'w', 'n', 'd', 'g', 'j', 'm', 'p', 's', 'v', 'r', 't', 'h', 'l', 'z', 'y', 'f', 'x']


Sentence case of "The quick brown dog jumps Over the lazy fox" is
 "The Quick Brown Dog Jumps Over The Lazy Fox"


Removing stop words from
"Rust is a multi-paradigm programming language focused on performance and safety, especially safe concurrency.[15][16] Rust is syntactically similar to C++,[17] but provides memory safety without using garbage collection.\nRust was originally designed by Graydon Hoare at Mozilla Research, with contributions from Dave Herman, Brendan Eich, and others.[18][19] The designers refined the language while writing the Servo layout or browser engine,[20] and the Rust compiler. The compiler is free and open-source software dual-licensed under the MIT License and Apache License 2.0."
gives
"Rust multi-paradigm programming language focused performance safety, especially safe concurrency.[15][16] Rust syntactically similar C++,[17] provides memory safety without using garbage collection.\nRust originally designed Graydon Hoare Mozilla Research, contributions Dave Herman, Brendan Eich, others.[18][19] designers refined language writing Servo layout browser engine,[20] Rust compiler. compiler free open-source software dual-licensed MIT License Apache License 2.0."




>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                                              LIB_MATRIX
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


This matrix will be used from here on
[23.0, 45.0, 12.0, 45.6, 218.0]
[3.0, 5.0, 12.5, 456.0, 28.1]
[13.0, 4.0, 12.7, 5.6, 128.0]
[73.0, 45.4, 120.0, 4.6, 8.0]
[27.0, 4.5, 1.0, 4.6, 8.0]


Calculating Determinant...
The determinant is : 7362542415.10779

Calculating Determinant...
Thats too much, the round off is : 7362542415.11

Is this a square matrix? true

The Inverse is :
[-0.003925035645689607, -0.00005307399509152724, 0.004259961677319762, -0.0003808564498380129, 0.03936511336552426]
[0.024595287306789476, -0.0019075609494864595, -0.041183831465310586, 0.002153913375012802, -0.006733881204985007]
[-0.006936097572655194, 0.0006761398874642079, 0.012508071174575558, 0.007810288568525409, -0.021305709861598535]
[-0.00007353646070534313, 0.0022017777482321805, -0.00037794173576373956, -0.0001861942543362484, 0.000503386240111152]
[0.00032144185562091993, -0.00009841192364656371, 0.007442342139525423, -0.0007954101300636407, -0.0016956827862046745]


The Transpose is :
[23.0, 3.0, 13.0, 73.0, 27.0]
[45.0, 5.0, 4.0, 45.4, 4.5]
[12.0, 12.5, 12.7, 120.0, 1.0]
[45.6, 456.0, 5.6, 4.6, 4.6]
[218.0, 28.1, 128.0, 8.0, 8.0]


This
[3, 3, 3, 3, 3]
[3, 3, 3, 3, 3]
[3, 3, 3, 3, 3]
[3, 3, 3, 3, 3]
[3, 3, 3, 3, 3]


Converted into floats
[3.0, 3.0, 3.0, 3.0, 3.0]
[3.0, 3.0, 3.0, 3.0, 3.0]
[3.0, 3.0, 3.0, 3.0, 3.0]
[3.0, 3.0, 3.0, 3.0, 3.0]
[3.0, 3.0, 3.0, 3.0, 3.0]


Into
[23.0, 45.0, 12.0, 45.6, 218.0]
[3.0, 5.0, 12.5, 456.0, 28.1]
[13.0, 4.0, 12.7, 5.6, 128.0]
[73.0, 45.4, 120.0, 4.6, 8.0]
[27.0, 4.5, 1.0, 4.6, 8.0]


Multiplication of 5x5 and 5x5
Output will be 5x5
is

[1030.8, 1030.8, 1030.8, 1030.8, 1030.8]
[1513.8, 1513.8, 1513.8, 1513.8, 1513.8]
[489.9, 489.9, 489.9, 489.9, 489.9]
[753.0, 753.0, 753.0, 753.0, 753.0]
[135.3, 135.3, 135.3, 135.3, 135.3]


[[23.0, 45.0, 12.0, 45.6, 218.0], [3.0, 5.0, 12.5, 456.0, 28.1], [13.0, 4.0, 12.7, 5.6, 128.0], [73.0, 45.4, 120.0, 4.6, 8.0], [27.0, 4.5, 1.0, 4.6, 8.0]]
element wise into
[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]] is
[69.0, 135.0, 36.0, 136.8, 654.0]
[9.0, 15.0, 37.5, 1368.0, 84.30000000000001]
[39.0, 12.0, 38.099999999999994, 16.799999999999997, 384.0]
[219.0, 136.2, 360.0, 13.799999999999999, 24.0]
[81.0, 13.5, 3.0, 13.799999999999999, 24.0]


[[23.0, 45.0, 12.0, 45.6, 218.0], [3.0, 5.0, 12.5, 456.0, 28.1], [13.0, 4.0, 12.7, 5.6, 128.0], [73.0, 45.4, 120.0, 4.6, 8.0], [27.0, 4.5, 1.0, 4.6, 8.0]]
element wise added
[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]] is
[26.0, 48.0, 15.0, 48.6, 221.0]
[6.0, 8.0, 15.5, 459.0, 31.1]
[16.0, 7.0, 15.7, 8.6, 131.0]
[76.0, 48.4, 123.0, 7.6, 11.0]
[30.0, 7.5, 4.0, 7.6, 11.0]


[[23.0, 45.0, 12.0, 45.6, 218.0], [3.0, 5.0, 12.5, 456.0, 28.1], [13.0, 4.0, 12.7, 5.6, 128.0], [73.0, 45.4, 120.0, 4.6, 8.0], [27.0, 4.5, 1.0, 4.6, 8.0]]
element wise subtracted
[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]] is
[20.0, 42.0, 9.0, 42.6, 215.0]
[0.0, 2.0, 9.5, 453.0, 25.1]
[10.0, 1.0, 9.7, 2.5999999999999996, 125.0]
[70.0, 42.4, 117.0, 1.5999999999999996, 5.0]
[24.0, 1.5, -2.0, 1.5999999999999996, 5.0]


[[23.0, 45.0, 12.0, 45.6, 218.0], [3.0, 5.0, 12.5, 456.0, 28.1], [13.0, 4.0, 12.7, 5.6, 128.0], [73.0, 45.4, 120.0, 4.6, 8.0], [27.0, 4.5, 1.0, 4.6, 8.0]]
element wise divided
[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]] is
[7.666666666666667, 15.0, 4.0, 15.200000000000001, 72.66666666666667]
[1.0, 1.6666666666666667, 4.166666666666667, 152.0, 9.366666666666667]
[4.333333333333333, 1.3333333333333333, 4.233333333333333, 1.8666666666666665, 42.666666666666664]
[24.333333333333332, 15.133333333333333, 40.0, 1.5333333333333332, 2.6666666666666665]
[9.0, 1.5, 0.3333333333333333, 1.5333333333333332, 2.6666666666666665]


>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  Vectors
This :['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
becomes with a few zeros :['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', '0', '0', '0', '0', '0']
or can also become :['0', '0', '0', '0', '0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
becomes a 4x3 matrix
['a', 'b', 'c']
['d', 'e', 'f']
['g', 'h', 'i']
['j', 'k', 'l']


This [3, 3, 3, 3, 3, 3, 3, 3, 3, 3] becomes [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0] + [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0] = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0] . [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0] = 60.0

[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0] * [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0] = [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]

[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0] - [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0] / [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0] = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]

Min and Max number in [23.0, 45.0, 12.0, 45.6, 218.0] is (12.0, 218.0)]n
The numbers and their frequency in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', '0', '0', '0', '0', '0'] is
{'0': 5, 'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1, 'h': 1, 'i': 1, 'j': 1, 'k': 1, 'l': 1}

Distinct numbers in  ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', '0', '0', '0', '0', '0'] is
['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', '0']


[23.0, 45.0, 12.0, 45.6, 218.0]
[3.0, 5.0, 12.5, 456.0, 28.1]
[13.0, 4.0, 12.7, 5.6, 128.0]
[73.0, 45.4, 120.0, 4.6, 8.0]
[27.0, 4.5, 1.0, 4.6, 8.0]


()
into
[3.0, 3.0, 3.0, 3.0, 3.0]
is
[1030.8, 1513.8, 489.9, 753.0, 135.3]


[3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
breaks in to
[3, 3]
[3, 3]
[3, 3]
[3, 3]
[3, 3]


or breaks in to
[3, 3, 3, 3, 3]
[3, 3, 3, 3, 3]



['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
splits at i
['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
['i', 'j', 'k', 'l']



[3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
splits at 3
[]
[3]
[3]
[3]
[3]
[3]
[3]
[3]
[3]
[3]
[3]


0 is a "i32"
0.0 is a "f64"
"0" is a "&str"
'0' is a "char"


A: [[1, 2], [3, 5]]
JOINED with
B: [[0, 1], [5, 7]]
Wide
[1, 2, 0, 1]
[3, 5, 5, 7]


Long
[1, 2]
[3, 5]
[0, 1]
[5, 7]


First 2 rows of long is
[1, 2]
[3, 5]


Last 2 rows of long is
[0, 1]
[5, 7]


Row to column and vice versa conversion
This

[1, 6, 11]
[2, 7, 12]
[3, 8, 13]
[4, 9, 14]
[5, 10, 15]


5x3 becomes
3x5
becomes

[1, 2, 3, 4, 5]
[6, 7, 8, 9, 10]
[11, 12, 13, 14, 15]


5x3 becomes
3x5
3x5 becomes
5x3
Which was originally

[1, 6, 11]
[2, 7, 12]
[3, 8, 13]
[4, 9, 14]
[5, 10, 15]

> String converted to &str
String matrix to &str matrix

["A", "B", "C"]
["D", "E", "F"]


>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                                              LIB_ML
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


The mean of [23.0, 45.0, 12.0, 45.6, 218.0]  is 68.72
The variance of [23.0, 45.0, 12.0, 45.6, 218.0]  is 28689.168
The covariance of [23.0, 45.0, 12.0, 45.6, 218.0] and [21.0, 4.0, 12.5, 32.6, 118.0] is 15097.327999999998
The equation of [23.0, 45.0, 12.0, 45.6, 218.0] and [21.0, 4.0, 12.5, 32.6, 118.0] is b0 = 1.4569303647983105, b1 = 0.5262379166938546

>>>>>>>>>>>>>>>>> Simple Linear Regression
========================================================================================================================================================
b0 = 11.731951871657754 and b1= 0.018827985739750495
RMSE: 2.080646271630258
Predicted is [11.92023172905526, 11.901403743315509, 11.93905971479501]
Original is [10.0, 9.0, 11.0]

Reading the file ...
Number of rows = 149
IRIS DATA:
["sepal_length", "sepal_width", "petal_length", "petal_width", "species\r"]
[["5.1", "3.5", "1.4", "0.2", "setosa\r"], ["4.9", "3.0", "1.4", "0.2", "setosa\r"], ["4.7", "3.2", "1.3", "0.2", "setosa\r"], ["4.6", "3.1", "1.5", "0.2", "setosa\r"], ["5.0", "3.6", "1.4", "0.2", "setosa\r"], ["5.4", "3.9", "1.7", "0.4", "setosa\r"], ["4.6", "3.4", "1.4", "0.3", "setosa\r"], ["5.0", "3.4", "1.5", "0.2", "setosa\r"], ["4.4", "2.9", "1.4", "0.2", "setosa\r"], ["4.9", "3.1", "1.5", "0.1", "setosa\r"], ["5.4", "3.7", "1.5", "0.2", "setosa\r"], ["4.8", "3.4", "1.6", "0.2", "setosa\r"], ["4.8", "3.0", "1.4", "0.1", "setosa\r"], ["4.3", "3.0", "1.1", "0.1", "setosa\r"], ["5.8", "4.0", "1.2", "0.2", "setosa\r"], ["5.7", "4.4", "1.5", "0.4", "setosa\r"], ["5.4", "3.9", "1.3", "0.4", "setosa\r"], ["5.1", "3.5", "1.4", "0.3", "setosa\r"], ["5.7", "3.8", "1.7", "0.3", "setosa\r"], ["5.1", "3.8", "1.5", "0.3", "setosa\r"], ["5.4", "3.4", "1.7", "0.2", "setosa\r"], ["5.1", "3.7", "1.5", "0.4", "setosa\r"], ["4.6", "3.6", "1.0", "0.2", "setosa\r"], ["5.1", "3.3", "1.7", "0.5", "setosa\r"], ["4.8", "3.4", "1.9", "0.2", "setosa\r"], ["5.0", "3.0", "1.6", "0.2", "setosa\r"], ["5.0", "3.4", "1.6", "0.4", "setosa\r"], ["5.2", "3.5", "1.5", "0.2", "setosa\r"], ["5.2", "3.4", "1.4", "0.2", "setosa\r"], ["4.7", "3.2", "1.6", "0.2", "setosa\r"], ["4.8", "3.1", "1.6", "0.2", "setosa\r"], ["5.4", "3.4", "1.5", "0.4", "setosa\r"], ["5.2", "4.1", "1.5", "0.1", "setosa\r"], ["5.5", "4.2", "1.4", "0.2", "setosa\r"], ["4.9", "3.1", "1.5", "0.1", "setosa\r"], ["5.0", "3.2", "1.2", "0.2", "setosa\r"], ["5.5", "3.5", "1.3", "0.2", "setosa\r"], ["4.9", "3.1", "1.5", "0.1", "setosa\r"], ["4.4", "3.0", "1.3", "0.2", "setosa\r"], ["5.1", "3.4", "1.5", "0.2", "setosa\r"], ["5.0", "3.5", "1.3", "0.3", "setosa\r"], ["4.5", "2.3", "1.3", "0.3", "setosa\r"], ["4.4", "3.2", "1.3", "0.2", "setosa\r"], ["5.0", "3.5", "1.6", "0.6", "setosa\r"], ["5.1", "3.8", "1.9", "0.4", "setosa\r"], ["4.8", "3.0", "1.4", "0.3", "setosa\r"], ["5.1", "3.8", "1.6", "0.2", "setosa\r"], ["4.6", "3.2", "1.4", "0.2", "setosa\r"], ["5.3", "3.7", "1.5", "0.2", "setosa\r"], ["5.0", "3.3", "1.4", "0.2", "setosa\r"], ["7.0", "3.2", "4.7", "1.4", "versicolor\r"], ["6.4", "3.2", "4.5", "1.5", "versicolor\r"], ["6.9", "3.1", "4.9", "1.5", "versicolor\r"], ["5.5", "2.3", "4.0", "1.3", "versicolor\r"], ["6.5", "2.8", "4.6", "1.5", "versicolor\r"], ["5.7", "2.8", "4.5", "1.3", "versicolor\r"], ["6.3", "3.3", "4.7", "1.6", "versicolor\r"], ["4.9", "2.4", "3.3", "1.0", "versicolor\r"], ["6.6", "2.9", "4.6", "1.3", "versicolor\r"], ["5.2", "2.7", "3.9", "1.4", "versicolor\r"], ["5.0", "2.0", "3.5", "1.0", "versicolor\r"], ["5.9", "3.0", "4.2", "1.5", "versicolor\r"], ["6.0", "2.2", "4.0", "1.0", "versicolor\r"], ["6.1", "2.9", "4.7", "1.4", "versicolor\r"], ["5.6", "2.9", "3.6", "1.3", "versicolor\r"], ["6.7", "3.1", "4.4", "1.4", "versicolor\r"], ["5.6", "3.0", "4.5", "1.5", "versicolor\r"], ["5.8", "2.7", "4.1", "1.0", "versicolor\r"], ["6.2", "2.2", "4.5", "1.5", "versicolor\r"], ["5.6", "2.5", "3.9", "1.1", "versicolor\r"], ["5.9", "3.2", "4.8", "1.8", "versicolor\r"], ["6.1", "2.8", "4.0", "1.3", "versicolor\r"], ["6.3", "2.5", "4.9", "1.5", "versicolor\r"], ["6.1", "2.8", "4.7", "1.2", "versicolor\r"], ["6.4", "2.9", "4.3", "1.3", "versicolor\r"], ["6.6", "3.0", "4.4", "1.4", "versicolor\r"], ["6.8", "2.8", "4.8", "1.4", "versicolor\r"], ["6.7", "3.0", "5.0", "1.7", "versicolor\r"], ["6.0", "2.9", "4.5", "1.5", "versicolor\r"], ["5.7", "2.6", "3.5", "1.0", "versicolor\r"], ["5.5", "2.4", "3.8", "1.1", "versicolor\r"], ["5.5", "2.4", "3.7", "1.0", "versicolor\r"], ["5.8", "2.7", "3.9", "1.2", "versicolor\r"], ["6.0", "2.7", "5.1", "1.6", "versicolor\r"], ["5.4", "3.0", "4.5", "1.5", "versicolor\r"], ["6.0", "3.4", "4.5", "1.6", "versicolor\r"], ["6.7", "3.1", "4.7", "1.5", "versicolor\r"], ["6.3", "2.3", "4.4", "1.3", "versicolor\r"], ["5.6", "3.0", "4.1", "1.3", "versicolor\r"], ["5.5", "2.5", "4.0", "1.3", "versicolor\r"], ["5.5", "2.6", "4.4", "1.2", "versicolor\r"], ["6.1", "3.0", "4.6", "1.4", "versicolor\r"], ["5.8", "2.6", "4.0", "1.2", "versicolor\r"], ["5.0", "2.3", "3.3", "1.0", "versicolor\r"], ["5.6", "2.7", "4.2", "1.3", "versicolor\r"], ["5.7", "3.0", "4.2", "1.2", "versicolor\r"], ["5.7", "2.9", "4.2", "1.3", "versicolor\r"], ["6.2", "2.9", "4.3", "1.3", "versicolor\r"], ["5.1", "2.5", "3.0", "1.1", "versicolor\r"], ["5.7", "2.8", "4.1", "1.3", "versicolor\r"], ["6.3", "3.3", "6.0", "2.5", "virginica\r"], ["5.8", "2.7", "5.1", "1.9", "virginica\r"], ["7.1", "3.0", "5.9", "2.1", "virginica\r"], ["6.3", "2.9", "5.6", "1.8", "virginica\r"], ["6.5", "3.0", "5.8", "2.2", "virginica\r"], ["7.6", "3.0", "6.6", "2.1", "virginica\r"], ["4.9", "2.5", "4.5", "1.7", "virginica\r"], ["7.3", "2.9", "6.3", "1.8", "virginica\r"], ["6.7", "2.5", "5.8", "1.8", "virginica\r"], ["7.2", "3.6", "6.1", "2.5", "virginica\r"], ["6.5", "3.2", "5.1", "2.0", "virginica\r"], ["6.4", "2.7", "5.3", "1.9", "virginica\r"], ["6.8", "3.0", "5.5", "2.1", "virginica\r"], ["5.7", "2.5", "5.0", "2.0", "virginica\r"], ["5.8", "2.8", "5.1", "2.4", "virginica\r"], ["6.4", "3.2", "5.3", "2.3", "virginica\r"], ["6.5", "3.0", "5.5", "1.8", "virginica\r"], ["7.7", "3.8", "6.7", "2.2", "virginica\r"], ["7.7", "2.6", "6.9", "2.3", "virginica\r"], ["6.0", "2.2", "5.0", "1.5", "virginica\r"], ["6.9", "3.2", "5.7", "2.3", "virginica\r"], ["5.6", "2.8", "4.9", "2.0", "virginica\r"], ["7.7", "2.8", "6.7", "2.0", "virginica\r"], ["6.3", "2.7", "4.9", "1.8", "virginica\r"], ["6.7", "3.3", "5.7", "2.1", "virginica\r"], ["7.2", "3.2", "6.0", "1.8", "virginica\r"], ["6.2", "2.8", "4.8", "1.8", "virginica\r"], ["6.1", "3.0", "4.9", "1.8", "virginica\r"], ["6.4", "2.8", "5.6", "2.1", "virginica\r"], ["7.2", "3.0", "5.8", "1.6", "virginica\r"], ["7.4", "2.8", "6.1", "1.9", "virginica\r"], ["7.9", "3.8", "6.4", "2.0", "virginica\r"], ["6.4", "2.8", "5.6", "2.2", "virginica\r"], ["6.3", "2.8", "5.1", "1.5", "virginica\r"], ["6.1", "2.6", "5.6", "1.4", "virginica\r"], ["7.7", "3.0", "6.1", "2.3", "virginica\r"], ["6.3", "3.4", "5.6", "2.4", "virginica\r"], ["6.4", "3.1", "5.5", "1.8", "virginica\r"], ["6.0", "3.0", "4.8", "1.8", "virginica\r"], ["6.9", "3.1", "5.4", "2.1", "virginica\r"], ["6.7", "3.1", "5.6", "2.4", "virginica\r"], ["6.9", "3.1", "5.1", "2.3", "virginica\r"], ["5.8", "2.7", "5.1", "1.9", "virginica\r"], ["6.8", "3.2", "5.9", "2.3", "virginica\r"], ["6.7", "3.3", "5.7", "2.5", "virginica\r"], ["6.7", "3.0", "5.2", "2.3", "virginica\r"], ["6.3", "2.5", "5.0", "1.9", "virginica\r"], ["6.5", "3.0", "5.2", "2.0", "virginica\r"], ["6.2", "3.4", "5.4", "2.3", "virginica\r"], ["5.9", "3.0", "5.1", "1.8", "virginica"]]


========================================================================================================================================================
["1", "6", "2", "6", "8", "2", "23", "3", "5", "2", "4", "2", "0"]
is now
[1.0, 6.0, 2.0, 6.0, 8.0, 2.0, 23.0, 3.0, 5.0, 2.0, 4.0, 2.0, 0.0]
with missing values at
[]
========================================================================================================================================================
Categorized ["1", "6", "2", "6", "8", "2", "23", "3", "5", "2", "4", "2", "0"] is now [1.0, 2.0, 3.0, 2.0, 4.0, 3.0, 5.0, 6.0, 7.0, 3.0, 8.0, 3.0, 9.0]
[1.0, 2.0, 3.0, -5.0, -7.0, 0.0]
Normalized to : [0.8, 0.9, 1.0, 0.19999999999999996, 0.0, 0.7]

In [1.0, 2.0, 3.0, -5.0, -7.0, 0.0]
minimum is -7 and maximum is 3

Shuffling [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] gives [3.0, 5.0, 7.0, 10.0, 4.0, 2.0, 1.0, 8.0, 6.0, 9.0]
Shuffling again [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] gives [3.0, 1.0, 2.0, 4.0, 5.0, 10.0, 7.0, 9.0, 6.0, 8.0]
Shuffling again [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] gives [8.0, 4.0, 5.0, 2.0, 3.0, 6.0, 10.0, 9.0, 1.0, 7.0]

Shuffling [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] gives [8.0, 5.0, 2.0, 9.0, 1.0, 10.0, 6.0, 7.0, 4.0, 3.0]
Shuffling again [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] gives [8.0, 2.0, 1.0, 9.0, 10.0, 6.0, 5.0, 3.0, 4.0, 7.0]
Shuffling again [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] gives [2.0, 7.0, 6.0, 5.0, 3.0, 9.0, 1.0, 4.0, 10.0, 8.0]

Train and test vectors of [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] with 50% test split is ([3.0, 7.0, 1.0, 8.0, 6.0], [4.0, 9.0, 10.0, 5.0, 2.0])
Train and test vectors of [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] with 20% test split is ([6.0, 10.0, 5.0, 4.0, 8.0, 1.0, 3.0, 9.0], [7.0, 2.0])

The original matrix
[1.0, 2.0, 3.0, 4.0, 5.0]
[3.0, 5.0, 8.0, 1.0, 0.3]
[0.5, 0.6, 0.1, 0.0, 0.0]
[0.0, 0.0, 0.0, 0.0, 0.0]
[1.0, 1.0, 1.0, 1.0, 1.0]


Shuffling gives
[1.0, 2.0, 3.0, 4.0, 5.0]
[3.0, 5.0, 8.0, 1.0, 0.3]
[0.5, 0.6, 0.1, 0.0, 0.0]
[1.0, 1.0, 1.0, 1.0, 1.0]
[0.0, 0.0, 0.0, 0.0, 0.0]


Shuffling again gives
[0.0, 0.0, 0.0, 0.0, 0.0]
[1.0, 2.0, 3.0, 4.0, 5.0]
[3.0, 5.0, 8.0, 1.0, 0.3]
[1.0, 1.0, 1.0, 1.0, 1.0]
[0.5, 0.6, 0.1, 0.0, 0.0]


Shuffling again gives
[0.0, 0.0, 0.0, 0.0, 0.0]
[1.0, 1.0, 1.0, 1.0, 1.0]
[1.0, 2.0, 3.0, 4.0, 5.0]
[0.5, 0.6, 0.1, 0.0, 0.0]
[3.0, 5.0, 8.0, 1.0, 0.3]


Train matrix with 50% test split is
[3.0, 5.0, 8.0, 1.0, 0.3]
[0.5, 0.6, 0.1, 0.0, 0.0]
[0.0, 0.0, 0.0, 0.0, 0.0]


Train matrix with 20% test split is
[1.0, 2.0, 3.0, 4.0, 5.0]
[1.0, 1.0, 1.0, 1.0, 1.0]
[3.0, 5.0, 8.0, 1.0, 0.3]
[0.0, 0.0, 0.0, 0.0, 0.0]


There is a strong negative correlation between the two :
Pearson Correlation of [56.0, 75.0, 45.0, 71.0, 62.0, 64.0, 58.0, 80.0, 76.0, 61.0] and [-66.0, -70.0, -40.0, -60.0, -65.0, -56.0, -59.0, -77.0, -67.0, -63.0] : -0.8058805796401135

Spearman's rank of [56.0, 75.0, 45.0, 71.0, 62.0, 64.0, 58.0, 80.0, 76.0, 61.0] : [(45.0, 1.0), (56.0, 2.0), (58.0, 3.0), (61.0, 4.0), (62.0, 5.0), (64.0, 6.0), (71.0, 7.0), (75.0, 8.0), (76.0, 9.0), (80.0, 10.0)]
There is a strong negative correlation between the two :
Spearman's Correlation of [56.0, 75.0, 45.0, 71.0, 62.0, 64.0, 58.0, 80.0, 76.0, 61.0] and [-66.0, -70.0, -40.0, -60.0, -65.0, -56.0, -59.0, -77.0, -67.0, -63.0] : -0.6727272727272726

Std. Dev of [56.0, 75.0, 45.0, 71.0, 62.0, 64.0, 58.0, 80.0, 76.0, 61.0] is 10.186265262597475

Position of 2 in [1, 1, 1, 1, 4, 2, 2, 5, 2, 7, 5, 3, 3, 6, 7, 5, 4, 0, 0, 10] is [5, 6, 8]

In
[0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0, 1.3, 1.4, 1.0, 1.5, 1.0, 1.4, 1.3, 1.4, 1.5, 1.0, 1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7, 1.5, 1.0]
[1.1, 1.0, 0.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3, 1.2, 1.4, 1.2, 1.0, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3, 2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8, 1.8, 2.5, 2.0, 1.9, 2.1, 2.0, 2.4, 2.3, 1.8, 2.2, 2.3, 1.5, 2.3, 2.0, 2.0, 1.8, 2.1, 1.8, 1.8, 1.8, 2.1, 1.6, 1.9, 2.0, 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9, 2.3, 2.5, 2.3, 1.9, 2.0, 2.3, 1.8]


0.2 can be found in [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 8), (0, 10), (0, 11), (0, 14), (0, 20), (0, 22), (0, 24), (0, 25), (0, 27), (0, 28), (0, 29), (0, 30), (0, 33), (0, 35), (0, 36), (0, 38), (0, 39), (0, 42), (0, 46), (0, 47), (0, 48), (0, 49), (1, 2)]

Z-score of 4 in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] is -0.35355339059327373

["A", "B", "C"]
One Hot Encoded
[1, 0, 0]
[0, 1, 0]
[0, 0, 1]



>>>>>>>>>>>>>>>>> ORDINARY LEAST SQUARE
Reading the file ...
Number of rows = 9567
The target here is header named: "PE"
"Training data" : 7655x5
"Testing data" : 1913x5
"Training data" : 7655x5
7655x5 becomes
5x7655
Features
[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
[6.28, 11.3, 20.99, 26.5, 10.36, 17.41]
[43.02, 43.14, 67.07, 66.05, 37.83, 40.55]
[1013.72, 1019.56, 1005.17, 1017.03, 1005.87, 1003.91]
[88.13, 99.83, 82.41, 61.34, 98.56, 76.87]


Multiplication of 5x7655 and 7655x5
Output will be 5x5
Multiplication of 5x5 and 5x1
Output will be 5x1


The coeficients of a columns as per simple linear regression on 20.0% of data is :
[("AT", -1.9723823959287756), ("V", -0.23430529857449756), ("AP", 0.07332101598694862), ("RH", -0.15708961379255015)] and b0 is : 443.0622340977134
RMSE : 4.676516060456772
MSE : 21.869802463710133
MAE : 3.693984317854811
MAPE : 0.8135031489132296
R2 and adjusted R2 : (0.9998950102848876, 0.999894735010333)

>>>>>>>>>>>>>>>>> Simple Logistic Regression
Reading the file ...
Number of rows = 1371
1098x5 becomes
5x1098
274x5 becomes
5x274
Using the actual values without preprocessing unless 's' or 'm' is passed
"Training features" : 4x1098
"Test features" : 4x274
Training target: 1098
Test target: 274
Reducing loss ...........................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................|------------------------|
|  159.0    |   2.0
|------------------------|
|  0.0    |   113.0
|------------------------|
Accuracy : 0.993
Precision : 0.988
Recall (sensitivity) : 1.000
Specificity: 0.983
F1 : 2.000


>>>>>>>>>>>>>>>>> K-Nearest Neighbour
Reading the file ...
Number of rows = 1371
Using the actual values without preprocessing unless 's' or 'm' is passed
"train Rows:" : 1098x4
"test Rows:" : 274x4


Calculating KNN using euclidean distance ...
Metrics

|------------------------|
|  135.0    |   3.0
|------------------------|
|  0.0    |   136.0
|------------------------|
Accuracy : 0.989
Precision : 0.978
Recall (sensitivity) : 1.000
Specificity: 0.978
F1 : 2.000
*/
