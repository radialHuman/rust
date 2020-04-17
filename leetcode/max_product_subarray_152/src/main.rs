fn main() {
    println!("{:?}", max_product(vec![4, 2, 0, 6, 7, 4, 1, -1, -7, 3, 2]));
}

fn max_product(v: Vec<i32>) -> i32 {
    let mut input_vector: Vec<i32> = vec![];
    let mut output_vector: Vec<&i32> = vec![];
    let mut output = v[0];
    for (n, i) in v.iter().enumerate() {
        if n != 0 && output * i > output {
            output_vector.push(i);
        } else {
            input_vector = input_vector[n..].to_vec();
        }
    }
    output
}
