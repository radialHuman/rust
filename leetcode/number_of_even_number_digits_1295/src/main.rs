// Given an array nums of integers, return how many of them contain an even number of digits.
/*
EXAMPLE 1
Input: nums = [12,345,2,6,7896]
Output: 2
Explanation:
12 contains 2 digits (even number of digits).
345 contains 3 digits (odd number of digits).
2 contains 1 digit (odd number of digits).
6 contains 1 digit (odd number of digits).
7896 contains 4 digits (even number of digits).
Therefore only 12 and 7896 contain an even number of digits.
*/

fn main() {
    println!("{}", find_numbers(vec![1, 12, 345, 43, 21]));
}

fn find_numbers(nums: Vec<i32>) -> i32 {
    let mut output = 0;
    for i in nums {
        let len = i.to_string().chars().count(); // getting length of a string
        if len % 2 == 0 {
            output += 1;
        }
    }
    output
}
