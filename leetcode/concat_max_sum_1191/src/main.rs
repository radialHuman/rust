fn main() {
    println!("{}", k_concatenation_max_sum(vec![-2, 1, 1], 5));
}
fn k_concatenation_max_sum(arr: Vec<i32>, k: i32) -> i32 {
    let mut arr_extended = arr.clone();
    let mut arr_split: Vec<Vec<i32>>;
    let mut output: i32 = arr.iter().sum();
    for _ in 1..k - 1 {
        arr_extended = [&arr_extended[..], &arr[..]].concat();
        println!("{:?}", arr_extended);
        // replace negative with 0
        println!("{:?}", arr_extended.iter().map(|w| *w * 0));
    }
    output
}
