// Given a 32-bit signed integer, reverse digits of an integer.

/*
EXAMPLE
Example 1:

Input: 123
Output: 321
Example 2:

Input: -123
Output: -321
Example 3:

Input: 120
Output: 21
*/

fn main() {
    println!("{:?}", reverse_int(1320));
}
fn reverse_int(n: i32) -> i32 {
    let mut output = 0;
    if n < 0 {
        output = n * -1;
        // reverse it
        let s = output.to_string().chars().rev().collect::<String>();
        output = s.parse::<i32>().unwrap() * -1;
        output
    } else {
        if n > 0 {
            n.to_string()
                .chars()
                .rev()
                .collect::<String>()
                .parse::<i32>()
                .unwrap()
        } else {
            0
        }
    }
}
