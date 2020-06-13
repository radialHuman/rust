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
    println!("'0' is a {:?}", type_of('0'));

    println!();
    println!();
    println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    println!("                                                              LIB_ML");
    println!(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    println!();
    println!();
}

/*
OUTPUT

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                                              LIB_NN
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Bias of 1.0 is introduced to all the neurons => [1.0, 1.0, 1.0, 1.0, 1.0]


Random weights introduced to all the neurons and input [10x5] =>
[0.31, -0.929, -0.723, 0.962, -0.129]
[-0.483, -0.423, 0.273, 0.686, 0.585]
[-0.751, 0.301, 0.225, -0.136, 0.436]
[-0.826, -0.591, 0.707, 0.261, 0.696]
[-0.782, 0.168, -0.739, 0.509, 0.441]
[-0.341, 0.865, -0.848, 0.684, -0.413]
[0.798, -0.968, -0.712, -0.843, 0.502]
[0.39, 0.228, 0.001, 0.526, -0.53]
[0.627, -0.433, 0.876, 0.774, -0.208]
[0.512, -0.79, -0.688, 0.305, -0.975]


(To check randomness) Random weights introduced to all the neurons and input [10x5] =>
[0.492, 0.326, 0.946, -0.209, 0.411]
[0.882, -0.046, 0.334, -0.547, -0.638]
[-0.669, 0.75, -0.552, 0.945, 0.461]
[0.24, 0.565, 0.003, 0.199, -0.355]
[-0.624, -0.866, 0.184, -0.778, -0.766]
[0.794, 0.846, -0.363, -0.251, -0.969]
[-0.88, -0.038, 0.821, 0.266, 0.815]
[0.556, -0.602, -0.425, -0.256, -0.58]
[0.609, -0.663, -0.928, 0.803, 0.094]
[0.549, 0.341, -0.579, -0.954, -0.287]


For input

[23.0, 45.0, 12.0, 45.6, 218.0, -12.7, -19.0, 2.0, 5.8, 2.0]
[3.0, 5.0, 12.5, 456.0, 28.1, -12.9, -19.2, 2.5, 8.0, 222.0]
[13.0, 4.0, 12.7, 5.6, 128.0, -12.1, -19.2, 15.2, 54.0, 32.0]
[73.0, 45.4, 120.0, 4.6, 8.0, -1.0, -19.2, 23.8, 10.0, 22.0]
[27.0, 4.5, 1.0, 4.6, 8.0, -1.0, -19.2, 2.7, 2.5, 12.0]


Weights

[0.718, 0.269, 0.046, -0.643, 0.328]
[0.235, 0.483, -0.889, -0.385, -0.273]
[-0.15, 0.677, -0.019, -0.195, -0.717]
[-0.292, -0.668, 0.989, -0.212, 0.882]
[0.074, 0.311, 0.68, 0.635, -0.712]
[0.948, 0.914, -0.379, -0.232, 0.365]
[-0.572, -0.745, 0.221, 0.94, 0.245]
[-0.206, -0.82, -0.049, -0.061, -0.321]
[-0.18, -0.862, 0.68, 0.572, -0.869]
[-0.704, -0.265, 0.941, 0.766, -0.756]


Bias
[0.0, 0.0, 0.0, 0.0, 0.0]


Using TanH
Multiplication of 5x10 and 10x5
Output will be 5x5
Alpha is for 'leaky relu' only, it is not taken into account here
The output of the layer is

[0.9999959054546264, -1.0, 1.0, 1.0, -0.9999999999613259]
[1.0, -0.38276902412982955, 1.0, 1.0, 1.0]
[1.0, -1.0, 1.0, -1.0, -0.9999999999999841]
[1.0, 1.0, 1.0, 1.0, 1.0]
[1.0, 1.0, 1.0, 1.0, 1.0]





Using Sigmoid
Multiplication of 5x10 and 10x5
Output will be 5x5
Alpha is for 'leaky relu' only, it is not taken into account here
The output of the layer is

[1.0, 0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000019724290608060442, 0.9999999999997484, 0.00000000000000000000000000000000000000000000000000000000000000000000000000006899522184219239, 0.000000022290616711712523]
[1.0, 1.0, 1.0, 0.9853130490804533, 1.0]
[0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002512690321639079, 1.0, 0.000000000000000000000000000000000000000000000000000000000000015805545486703512, 0.000000000000000000000000000000000000000000000000005188474600451208, 0.0000000000000006108379321927211]
[1.0, 1.0, 1.0, 1.0, 1.0]
[0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000022082368515788063, 0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000933022351146738, 0.00000000000000000000000000000000000000000000002567157795339279, 0.00000000000000000000000000000000000000000000000000000000000000000000000000000000003314022533624378, 0.00000000011616610608020281]





Using ReLU
Multiplication of 5x10 and 10x5
Output will be 5x5
Alpha is for 'leaky relu' only, it is not taken into account here
The output of the layer is

[0.0, 118.40380000000002, 0.0, 100.0908, 0.0]
[0.0, 0.0, 0.0, 10.104999999999997, 0.0]
[0.0, 0.0, 0.0, 84.85259999999998, 36.378499999999995]
[0.0, 0.0, 0.0, 0.0, 0.0]
[138.21230000000003, 616.7357999999999, 86.40639999999999, 49.5946, 0.0]





Using Leaky ReLU
Multiplication of 5x10 and 10x5
Output will be 5x5
The output of the layer is

[161.53289999999996, -38.67325000000001, 168.61819999999997, 68.8744, 21.237600000000004]
[72.64529999999999, -58.94013, -1.9525300000000005, 90.4246, 1.6426999999999996]
[171.7342, 46.557599999999994, 118.88179999999998, 37.75500000000001, 40.978699999999996]
[-15.421240000000004, -14.532939999999996, -9.171109999999999, -4.766760000000002, -1.2913000000000001]
[-3.300990000000001, -4.1204600000000005, -4.560379999999999, 43.89519999999999, -0.13778000000000043]


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

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                                              LIB_ML
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


*/
