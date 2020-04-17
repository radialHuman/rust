fn main() {
    println!("{}", buddy_strings("abab".to_string(), "abab".to_string()));
}

fn buddy_strings(a: String, b: String) -> bool {
    let mut output: bool;
    // if empty string
    if a == "" || b == "" {
        output = false;
    }
    // if they are the same words but 2 letter word
    else if a == b && a.len() == 2 {
        // if characters are same
        let list_a: Vec<char> = a.chars().collect();
        let list_b: Vec<char> = b.chars().collect();
        if list_a[0] != list_a[1] && list_a[0] != list_b[1] {
            output = false;
        } else {
            output = true;
        }
    }
    // if same but more than 2 letters
    else if a == b && a.len() != 2 {
        output = true;
    }
    // check the length if not the same words
    else if a.len() == b.len() {
        // create vector
        let list_a: Vec<char> = a.chars().collect();
        let list_b: Vec<char> = b.chars().collect();
        let mut left_out_a: Vec<char> = vec![];
        let mut left_out_b: Vec<char> = vec![];
        // removing common words
        for i in 0..list_a.len() {
            if list_a[i] != list_b[i] {
                left_out_a.push(list_a[i]);
                left_out_b.push(list_b[i]);
            }
        }
        println!("{:?}{:?}", left_out_a, left_out_b);
        // checking if the left outs are the same and just 2
        if left_out_a.len() == 2 && left_out_b.len() == 2 {
            let mut count: Vec<&char> = vec![];
            for i in left_out_a.iter() {
                if left_out_b.contains(i) {
                    count.push(i);
                }
            }
            println!("{:?}", count);
            if count.len() == 2 {
                output = true;
            }
        }
    }
    output
}
