// 315. Count of Smaller Numbers After Self
// You are given an integer array nums and you have to return a new counts array. The counts array has the property where counts[i] is the number of smaller elements to the right of nums[i].

/*
EXAMPLE
Input: [5,2,6,1]
Output: [2,1,1,0]
Explanation:
To the right of 5 there are 2 smaller elements (2 and 1).
To the right of 2 there is only 1 smaller element (1).
To the right of 6 there is 1 smaller element (1).
To the right of 1 there is 0 smaller element.
*/

fn main() {
    println!("{:?}", count_smaller(vec![1, 7, 3, 2, 4, 8, 0, 9, -1]));
}

fn count_smaller(nums: Vec<i32>) -> Vec<i32> {
    let mut output = vec![];
    for (n, i) in nums.iter().enumerate() {
        let mut smaller = 0;
        for j in nums[n..].iter() {
            // to make it look only after the current value and not in the whole array
            if i > j {
                smaller += 1;
            }
        }
        output.push(smaller);
    }
    output
}
