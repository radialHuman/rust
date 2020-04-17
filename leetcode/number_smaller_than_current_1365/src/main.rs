// Given the array nums, for each nums[i] find out how many numbers in the array are smaller than it. That is, for each nums[i] you have to count the number of valid j's such that j != i and nums[j] < nums[i].
// Return the answer in an array.

/*
EXAMPLE
Input: nums = [8,1,2,2,3]
Output: [4,0,1,1,3]
Explanation:
For nums[0]=8 there exist four smaller numbers than it (1, 2, 2 and 3).
For nums[1]=1 does not exist any smaller number than it.
For nums[2]=2 there exist one smaller number than it (1).
For nums[3]=2 there exist one smaller number than it (1).
For nums[4]=3 there exist three smaller numbers than it (1, 2 and 2).
*/

fn main() {
    println!(
        "{:?}",
        smaller_numbers_than_current(vec![1, 5, 2, 5, 3, 6, 8, 4, 0, 4, 7, 9, 5, 3, 3])
    );
}

fn smaller_numbers_than_current(nums: Vec<i32>) -> Vec<i32> {
    let mut output = vec![];
    for i in nums.iter() {
        let mut smaller_than_current = 0;
        for j in nums.iter() {
            if i > j {
                smaller_than_current += 1;
            }
        }
        output.push(smaller_than_current);
    }
    output
}
