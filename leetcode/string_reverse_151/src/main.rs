// 151. Reverse Words in a String

/*Given an input string, reverse the string word by word.

Example 1:

Input: "  hello world!  "
Output: "world! hello"
Explanation: Your reversed string should not contain leading or trailing spaces.
*/

fn main() {
    reverse_words("a good   example".to_string());
}

fn reverse_words(s: String) -> String {
    let sa: Vec<&str> = s.split(" ").collect();
    let reverse = sa
        .iter()
        .rev()
        .filter(|x| x != &&"")
        .cloned() // or else error due to datatype
        .collect::<Vec<&str>>();
    reverse.join(" ").to_string()
}
