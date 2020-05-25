use simple_ml::*;

fn main() {
    println!(
        "{:?}",
        activation_relu(&vec![1., 3., 5., 6., 8., -3., -5., -7.])
    );

    // println!("Numerical {:?}", is_numerical(23.67));

    let mut v = vec![1., 3., 5., 6., 8., -3., -5., -7.];
    // let (min, max) = min_max_float(&v);
    // println!("min is {:?}\nmax is {:?}", min, max);

    // println!("{:?}\nNormalized to \n{:?}", v, normalize(&v));

    let v1 = vec![
        5, 6, 8, 4, 2, 3, 5, 9, 7, 4, 1, 2, 35, 6, 45, 48, 4, 21, 6, 13, 2168, 1, 5, 68, 1, -45, 0,
    ];
    // let (min, max) = min_max(&v1);
    // println!("min is {:?}\nmax is {:?}", min, max);
}
