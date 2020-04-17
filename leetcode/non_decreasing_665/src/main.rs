fn main() {
    let mut v = vec![1, 2, 3];
    println!("{:?}", can_it_be_non_decreasing(&mut v));
    let mut v = vec![4, 2, 3];
    println!("{:?}", can_it_be_non_decreasing(&mut v));
    let mut v = vec![4, 2, 3, 1];
    println!("{:?}", can_it_be_non_decreasing(&mut v));
    let mut v = vec![3, 4, 2, 3];
    println!("{:?}", can_it_be_non_decreasing(&mut v));
}

fn can_it_be_non_decreasing(v: &mut Vec<i32>) -> bool {
    // check the sum of differecne
    let difference = make_difference(&v);
    match difference.iter().filter(|x| x.is_negative()).count() {
        0 => {
            println!("{:?} is already non decreasing", v);
            return true;
        }
        1 => {
            println!("{:?} is can be non decreasing", v);
            make_this_non_decreasing(&v, &difference);
            return true;
        }
        _ => {
            println!("{:?} is cant be non decreasing", v);
            return false;
        }
    }
}

fn make_difference(v: &Vec<i32>) -> Vec<i32> {
    let mut difference = vec![0];
    for (n, _) in v.iter().enumerate() {
        if n + 1 < v.len() {
            // to avoid out of bound error
            difference.push(v[n + 1] - v[n]);
        }
    }
    difference
}

fn make_this_non_decreasing(v: &Vec<i32>, difference: &Vec<i32>) {
    let mut y = v.clone();
    println!("WAS {:?},{:?}", v, difference);
    for (n, i) in difference.iter().enumerate() {
        if i < &1 && n + 1 < v.len() {
            y[n] = y[n + 1] - 1;
        }
    }
    println!("is {:?},{:?}", y, make_difference(&y));
}
