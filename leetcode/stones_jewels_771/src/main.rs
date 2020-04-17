// https://leetcode.com/problems/jewels-and-stones/
/*
You're given strings J representing the types of stones that are jewels, and S representing the stones you have.  Each character in S is a type of stone you have.  You want to know how many of the stones you have are also jewels.

The letters in J are guaranteed distinct, and all characters in J and S are letters. Letters are case sensitive, so "a" is considered a different type of stone from "A".

Example 1:

Input: J = "aA", S = "aAAbbbb"
Output: 3
Example 2:

Input: J = "z", S = "ZZ"
Output: 0
*/
fn main() {
    println!(
        "{}",
        num_jewels_in_stones("Aa".to_string(), "aAAsde".to_string())
    );
}

pub fn num_jewels_in_stones(j: String, s: String) -> i32 {
    let output: i32;
    let a_j: Vec<char> = j.chars().collect();
    let a_s: Vec<char> = s.chars().collect();
    let mut intersection: Vec<bool> = vec![];
    for i in a_s {
        intersection.push(a_j.contains(&i));
    }
    output = intersection
        .into_iter()
        .filter(|&i| i == true)
        .collect::<Vec<_>>()
        .len() as i32;
    output
}
