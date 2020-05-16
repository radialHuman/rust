/*
To have a matrix_library of all the functions related to nn in one palce
Derived from nnfs
May turn into a crate with generics
*/
mod matrix_lib;

fn main() {
    // ReLU check
    let mut v1 = vec![1., 2., 5., 8., 3., 7., 5., -2., -3., -5.];
    println!("ReLU of {:?} is {:?}", v1, matrix_lib::activation_relu(&v1));

    let mut v2 = vec![1, 6, 2, 7, 3, 7, 99, 6, 3, 6, -12, -34];
    println!("ReLU of {:?} is {:?}", v2, matrix_lib::activation_relu(&v2));

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
}

/* OUTPUT
ReLU of [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, -2.0, -3.0, -5.0] is [1.0, 2.0, 5.0, 8.0, 3.0, 7.0, 5.0, 0.0, 0.0, 0.0]
ReLU of [1, 6, 2, 7, 3, 7, 99, 6, 3, 6, -12, -34] is [1, 6, 2, 7, 3, 7, 99, 6, 3, 6, 0, 0]
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
*/
