fn main() {
    let number = 6;
    let mut side1: Vec<u8> = vec![];
    let mut side2: Vec<u8> = vec![];
    let mut middle: Vec<u8> = vec![];
    // first line
    for _ in 0..number {
        side1.push(1);
        side2.push(1);
    }
    // middle lines
    for _ in 0..number - 2 {
        for (n, _) in side1.iter().enumerate() {
            if n == 0 || n == number - 1 {
                middle.push(1);
            } else {
                middle.push(0);
            }
        }
    }
    let output = shape_changer(&[[side1, middle].concat(), side2].concat(), number, number);
    print_a_matrix(&output);
}

pub fn shape_changer<T: std::clone::Clone>(
    list: &Vec<T>,
    columns: usize,
    rows: usize,
) -> Vec<Vec<T>> {
    /*Changes a list to desired shape matrix*/
    // println!("{},{}", &columns, &rows);
    let mut l = list.clone();
    let mut output = vec![vec![]; rows];
    for i in 0..rows {
        output[i] = l[..columns].iter().cloned().collect();
        // remove the ones pushed to putput
        l = l[columns..].iter().cloned().collect();
    }
    output
}

pub fn print_a_matrix<T: std::fmt::Debug>(matrix: &Vec<Vec<T>>) {
    for i in matrix.iter() {
        println!("{:?}", i);
    }
    println!("");
    println!("");
}
