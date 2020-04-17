extern crate map_vec; // for having set

fn main() {
    // LOGIC 1 (TOo slow)
    // println!("{}", subarrays_div_by_k(vec![4, 5, 0, -2, -3, 1], 5));
    // println!("{:?}", create_subarrays(&vec![4, 5, 0, -2, -3, 1]));

    // LOGIC 2
    println!("{}", subarrays_div_by_k_faster(vec![4, 5, 0, -2, -3, 1], 5));
}

fn subarrays_div_by_k_faster(a: Vec<i32>, k: i32) -> usize {
    let mut divisible = map_vec::Set::new();
    // take the array, have a moving window function from 1 to length if divisible add
    let mut b = &a;
    for j in 1..b.iter().len() {
        let b = &a[..j];
        // forward
        for i in 1..b.iter().len() + 1 {
            // println!("{:?}", &b[..i]);
            if &b[..i].iter().sum() % k == 0 {
                divisible.insert(&b[..i]);
            }
        }
        // backward
        for i in 0..b.iter().len() {
            // println!("{:?}", &b[i..]);
            if &b[i..].iter().sum() % k == 0 {
                divisible.insert(&b[i..]);
            }
        }
    }
    // println!("{:?}", divisible);
    // taking care of the entire vec, due to lack of it in the loops above
    if &a.iter().sum() % k == 0 {
        divisible.len() + 1
    } else {
        divisible.len()
    }
}

fn subarrays_div_by_k(a: Vec<i32>, k: i32) -> i32 {
    let mut output = 0;
    let x = create_subarrays(&a);
    for i in x {
        if &i.iter().sum() % k == 0 {
            output += 1;
            // println!("{:?}", i);
        }
    }
    output
}

fn create_subarrays(a: &Vec<i32>) -> Vec<Vec<i32>> {
    let mut output = vec![];
    for i in 0..a.len() + 1 {
        let b = &a[..i];
        for (n, _) in b.iter().enumerate() {
            output.push(b[n..].to_vec());
        }
    }
    output
}
