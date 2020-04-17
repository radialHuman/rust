fn main() {
    println!("{:?}", lucky_numbers(vec![vec![7, 8], vec![1, 2]]));
}

fn lucky_numbers(matrix: Vec<Vec<i32>>) -> Vec<i32> {
    let SIZE = matrix.len();
    let mut result = vec![];
    let mut positions: Vec<i32> = vec![];
    let mut minimumns: Vec<Option<&i32>> = vec![];
    for i in matrix.iter() {
        // finding minimum in each row
        minimumns.push(i.iter().min());
    }
    // finding the position of each minimum and if its the same
    for (_, i) in matrix.iter().enumerate() {
        for (m, j) in i.iter().enumerate() {
            for (_, k) in minimumns.iter().enumerate() {
                match k {
                    Some(x) => {
                        if *x == j {
                            positions.push(m as i32);
                        }
                    }
                    None => (),
                };
            }
        }
    }
    // println!("{:?}", positions);
    // println!("{:?}", minimumns);
    // finding if they are the max in their columns
    let mut subset: Vec<i32> = vec![];
    for (_, i) in positions.iter().enumerate() {
        for (_, j) in matrix.iter().enumerate() {
            for (o, _) in j.iter().enumerate() {
                if o as i32 == *i {
                    subset.push(j[o]);
                } else {
                    {}
                }
            }
        }
    }
    // println!("{:?}", subset);
    // checking if they are maximum in the group
    let mut l = 0;
    for i in minimumns {
        match i {
            Some(x) => {
                if Some(x) == subset[l..l + SIZE].to_vec().iter().max() {
                    result.push(*x)
                }
            }
            _ => (),
        };
        l += SIZE;
    }

    result
}
