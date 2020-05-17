/*
To have a matrix_library and neural_network_library of all the functions related to nn in one palce
Derived from nnfs
May turn into a crate with generics
*/
mod matrix_lib;
mod nn_lib;
use nn_lib::LayerDetails;

use rand::*;

fn main() {
    // matrix_lib.rs

    let v1 = vec![1., 2., 5., 8., 3., 7., 5., -2., -3., -5.];
    let mut v2 = vec![1, 6, 2, 7, 3, 7, 99, 6, 3, 6, -12, -34];

    // vector addition check
    let v1_plus_v2 =
        matrix_lib::vector_addition(&mut v1.iter().map(|x| *x as i32).collect(), &mut v2);
    println!(
        "The addition of of {:?} with {:?} is: {:?}",
        v1, v2, v1_plus_v2
    );

    // elemnt wise multiplicaiton check
    let v1_into_v2 = matrix_lib::element_wise_multiplication(
        &mut v1.iter().map(|x| *x as i32).collect(),
        &mut v2,
    );
    println!(
        "The multiplication of {:?} with {:?} is: {:?}",
        v1, v2, v1_into_v2
    );

    // Shape changer
    let v2 = matrix_lib::shape_changer(&v2, 4, 3);
    matrix_lib::print_a_matrix("\n2x5 version is:", &v2);

    // matrix transpose check
    let v2_t = matrix_lib::transpose(&v2);
    matrix_lib::print_a_matrix("The previous matrix is transposed to", &v2_t);

    // matrix multiplcation and dot product
    let v3 = vec![vec![1, 4, 4], vec![5, 8, 9], vec![0, 1, 6]];
    let v4 = vec![vec![1, 4, 4, 5], vec![5, 8, 9, 1], vec![0, 1, 6, 0]];
    let v3_v4 = matrix_lib::matrix_product(&v3, &v4);
    matrix_lib::print_a_matrix(
        &format!("The multiplicaiton of {:?} and {:?}", v3, v4),
        &v3_v4,
    );

    //================================================================================================================

    // nn_lib.rs
    // takes in only f64

    // ACTIVATION FUCNTION ReLU
    println!("ReLU of {:?} is {:?}", v1, nn_lib::activation_relu(&v1));

    // ACTIVATION FUNCTION Leaky ReLU
    println!(
        "Leaky ReLU of {:?} with alpha 0.1 is {:?}",
        v1,
        nn_lib::activation_leaky_relu(&v1, 0.1)
    );

    // ACTIVATION FUNCTION Sigmoid
    println!(
        "Sigmoid of {:?} is {:?}",
        v1,
        nn_lib::activation_sigmoid(&v1)
    );

    // ACTIVATION FUNCTION TanH
    println!("TanH of {:?} is {:?}", v1, nn_lib::activation_tanh(&v1));

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
    matrix_lib::print_a_matrix("\nInput generated is :", &input);

    // Output of a layer activation_function(input*weights+bias)
    let output = layer_1.output_of_layer(
        &input,
        &layer_1.create_weights(),
        &mut layer_1.create_bias(),
        nn_lib::activation_relu, // to be choosen by user, when there are other fucntiosn
    );

    matrix_lib::print_a_matrix("\nOutput generated is :", &output);
}

// fn main() {
//     let v1 = vec![1., 2., 5., 0.8, 3., 7., 5., -2., -3., -5.];
//     println!("Tanh of {:?} is {:?}", v1, nn_lib::activation_tanh(&v1));
// }

/* OUTPUT
The changed vector is [1, 2, 5, 8, 3, 7, 5, -2, -3, -5, 0, 0]
The addition of of [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, -2.0, -3.0, -5.0] with [1, 6, 2, 7, 3, 7, 99, 6, 3, 6, -12, -34] is: [2, 8, 7, 15, 6, 14, 104, 4, 0, 1]
The changed vector is [1, 2, 5, 8, 3, 7, 5, -2, -3, -5, 0, 0]
The multiplication of [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, -2.0, -3.0, -5.0] with [1, 6, 2, 7, 3, 7, 99, 6, 3, 6, -12, -34] is: [1, 12, 10, 56, 9, 49, 495, -12, -9, -30, 0, 0]

2x5 version is:
[1, 6, 2, 7]
[3, 7, 99, 6]
[3, 6, -12, -34]


The previous matrix is transposed to
[1, 3, 3]
[6, 7, 6]
[2, 99, -12]
[7, 6, -34]


Multiplication of 3x3 and 3x4
Output will be 3x4
The multiplicaiton of [[1, 4, 4], [5, 8, 9], [0, 1, 6]] and [[1, 4, 4, 5], [5, 8, 9, 1], [0, 1, 6, 0]]
[21, 40, 64]
[9, 45, 93]
[146, 33, 5]
[14, 45, 1]


ReLU of [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, -2.0, -3.0, -5.0] is [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, 0.0, 0.0, 0.0]
ReLU of [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, -2.0, -3.0, -5.0] is [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, 0.0, 0.0, 0.0]
Leaky ReLU of [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, -2.0, -3.0, -5.0] with alpha 0.1 is [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, -0.2, -0.30000000000000004, -0.5]
Sigmoid of [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, -2.0, -3.0, -5.0] is [0.2689414213699951, 0.11920292202211755, 0.0066928509242848554, 0.0003353501304664781, 0.04742587317756678, 0.0009110511944006454, 0.0066928509242848554, 0.8807970779778823, 0.9525741268224334, 0.9933071490757153]
Tanh of [1.0, 2.0, 5.0, 0.8, 3.0, 7.0, 5.0, -2.0, -3.0, -5.0] is [0.7615941559557649, 0.964027580075817, 0.999909204262595, 0.664036770267849, 0.9950547536867306, 0.9999983369439447, 0.999909204262595, -0.964027580075817, -0.9950547536867306, -0.999909204262595]

Input generated is :
[2.841796917427054, -9.642429895883167, 5.947542422193642, -0.7089463523840234, 5.859383897766733, -6.2538797189982365]
[2.2201573556313914, 4.0049163322786985, -7.068329918427283, 0.8999510383122242, -0.2718192044238421, 7.89790098253302]
[-0.45873814627376497, -8.313963733595017, 4.647970978803544, -2.8400881247252796, -6.68421775904239, 7.474504357284992]
[2.4815571525065216, -2.670196786758785, 1.5061039165492787, -3.0484532645171614, 4.620238827421392, -8.235417513010743]
[-2.7246071051242815, -9.992514124315527, 6.851401711772375, 2.244466469548163, -5.299456686549178, -9.609509335297037]
[-4.158851348469872, 1.777511251673257, -2.3634775508152606, 8.4813131683811, -2.9404141150866936, 5.2184755407380035]
[4.050054821930891, 1.6982073973273337, 3.335283775874597, 2.952928032927513, -7.04092565467886, 5.067140306554938]
[-7.1729602055128705, -2.902425548041223, 3.519077344902586, -2.5096059903988754, 2.2073149011405047, 7.430895104282264]
[6.594649454666843, 0.3302385583547096, 0.5810093774984271, 4.921223527624189, 7.397694934900482, -1.7619584456072488]
[-8.454636884206268, -0.3745556243006831, -1.340051006747709, 8.084840612210705, -8.319070417816029, -3.993595565033754]


Multiplication of 10x6 and 6x5
Output will be 10x5

Output generated is :
[0.0, 0.0, 0.0, 0.0, 0.0]
[7.0004096061045615, 0.0, 8.404835747765132, 4.315913662909908, 8.054086945853618]
[0.0, 2.815947186139281, 0.0, 0.0, 0.0]
[0.0, 0.0, 0.0, 5.456520688184097, 5.065717203744614]
[16.09474371221213, 6.7285083862362125, 11.982825756630756, 3.7671945003772755, 11.177223371417705]
[1.2615601031989527, 0.74625689383674, 1.2200495406310354, 1.6764591050705064, 2.830294390848986]
[0.0, 0.8415575414891734, 4.346076738995668, 0.0, 4.68225538889613]
[6.090658765294076, 0.0, 0.0, 12.161294763943598, 0.0]
[9.17106917474336, 0.0, 8.78681039685812, 0.0, 5.183524348938326]
[0.0, 2.9913842955020726, 1.730542570147415, 0.7245254298249522, 0.0]
*/
