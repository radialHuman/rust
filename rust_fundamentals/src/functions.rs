pub fn function() {
    // FUNCTIONS
    // METHODS
    // CLOUSERS
    let a_variable = func; // assining function to a variable
    a_variable(); // vairable acting as a functions

    // inline variable function
    let b_variable = |x: i32| -> f64 { x as f64 * 0.1 };
    println!("{}", b_variable(23));

    // HIGHER ORDER FUNCTIONS
    // task : if square of number is even add it
    let limit = 1000;
    let mut sum = 0;
    for i in 1..100 {
        if i * i % 2 == 0 {
            if i * i < limit {
                sum += i * i;
            }
        }
    }
    println!("The total SoS is {}", sum);

    // the above task can be performed in one statement
    let higher_sum: i32 = (1..100)
        .map(|x| x * x)
        .filter(|x| (x < &limit) && (x % 2 == 0))
        .into_iter()
        .sum::<i32>();
    println!(
        "The total SoS USING HIGHER ORDER FUNCTION is {:?}",
        higher_sum
    );
}

fn func() {
    println!("This is a variable used to call a function",);
}
