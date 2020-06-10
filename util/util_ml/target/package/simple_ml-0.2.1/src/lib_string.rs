/*
DESCRIPTION
-----------------------------------------
STRUCTS
-------
1. StringToMatch :
        > compare_percentage : comparision based on presence of characters and its position
            x calculate
        > clean_string : lower it and keep alphaneumericals only
            x char_vector
        > compare_chars
        > compare_position
        > fuzzy_subset : scores based on chuncks of string
            x n_gram
        > split_alpha_numericals : seperates numbers from the rest
        > char_count : Returns dictioanry of characters arranged in alphabetically increasing order with their frequency
        > frequent_char : Returns the more frequently occuring character in the string passed
        > char_replace : Finds a character, replaces it with a string at all positions or at just the first depending on operation argument

FUNCTIONS
---------
1. ...

*/

pub struct StringToMatch {
    pub string1: String,
    pub string2: String,
}

impl StringToMatch {
    pub fn compare_percentage(
        &self,
        weightage_for_position: f64,
        weightage_for_presence: f64,
    ) -> f64 {
        /*
            Scores by comparing characters and its position as per weightage passed
            Weightage passed as ratio
            ex: 2.,1. will give double weightage to position than presence
        */

        ((StringToMatch::compare_chars(&self) * weightage_for_presence * 100.)
            + (StringToMatch::compare_position(&self) * weightage_for_position * 100.))
            / 2.
    }

    pub fn clean_string(s1: String) -> String {
        /*
            Lowercase and removes special characters
        */

        // case uniformity
        let mut this = s1.to_lowercase();

        // only alpha neurmericals accents - bytes between 48-57 ,97-122, 128-201
        // https://www.utf8-chartable.de/unicode-utf8-table.pl?number=1024&utf8=dec&unicodeinhtml=dec
        let this_byte: Vec<_> = this
            .as_bytes()
            .iter()
            .filter(|a| {
                (**a > 47 && **a < 58) || (**a > 96 && **a < 123) || (**a > 127 && **a < 201)
            })
            .map(|a| *a)
            .collect();
        let new_this = std::str::from_utf8(&this_byte[..]).unwrap();
        new_this.to_string()
    }

    fn char_vector(String1: String) -> Vec<char> {
        /*
            String to vector of characters
        */
        let string1 = StringToMatch::clean_string(String1.clone());
        string1.chars().collect()
    }

    fn calculate(actual: f64, v1: &Vec<char>, v2: &Vec<char>) -> f64 {
        /*
            normalizes score by dividing it with the longest string's length
        */
        let larger = if v1.len() > v2.len() {
            v1.len()
        } else {
            v2.len()
        };
        (actual / larger as f64)
    }

    pub fn compare_chars(&self) -> f64 {
        /*
            Scores as per occurance of characters
        */
        let mut output = 0.;
        println!("{:?} vs {:?}", self.string1, self.string2);
        let vec1 = StringToMatch::char_vector(self.string1.clone());
        let vec2 = StringToMatch::char_vector(self.string2.clone());

        for i in vec1.iter() {
            if vec2.contains(i) {
                output += 1.;
            }
        }
        StringToMatch::calculate(output, &vec1, &vec2)
    }
    pub fn compare_position(&self) -> f64 {
        /*
            Scores as per similar positioning of characters
        */
        let mut output = 0.;
        println!("{:?} vs {:?}", self.string1, self.string2);
        let vec1 = StringToMatch::char_vector(self.string1.clone());
        let vec2 = StringToMatch::char_vector(self.string2.clone());

        let combined: Vec<_> = vec1.iter().zip(vec2.iter()).collect();

        for (i, j) in combined.iter() {
            if i == j {
                output += 1.;
            }
        }
        StringToMatch::calculate(output, &vec1, &vec2)
    }

    pub fn fuzzy_subset(&self, n_gram: usize) -> f64 {
        /*
            break into chuncks and compare if not a subset
        */
        let mut match_percentage = 0.;
        let vec1 = StringToMatch::clean_string(self.string1.clone());
        let vec2 = StringToMatch::clean_string(self.string2.clone());

        // finding the subset out of the two parameters
        let mut subset = vec2.clone();
        let mut superset = vec1.clone();
        if vec1.len() < vec2.len() {
            subset = vec1;
            superset = vec2;
        }

        let mut chunck_match_count = 0.;

        // whole string
        if superset.contains(&subset) {
            match_percentage = 100.
        } else {
            // breaking them into continous chuncks
            let superset_n = StringToMatch::n_gram(&superset, n_gram);
            let subset_n = StringToMatch::n_gram(&subset, n_gram);
            for i in subset_n.iter() {
                if superset_n.contains(i) {
                    chunck_match_count += 1.;
                }
            }
            // calculating match score
            let smaller = if superset_n.len() < subset_n.len() {
                superset_n.len()
            } else {
                subset_n.len()
            };
            match_percentage = (chunck_match_count / smaller as f64) * 100.
        }

        println!("{:?} in {:?}", subset, superset);
        match_percentage
    }

    fn n_gram<'a>(string: &'a str, window_size: usize) -> Vec<&'a str> {
        let vector: Vec<_> = string.chars().collect();
        let mut output = vec![];
        for (mut n, _) in vector.iter().enumerate() {
            while n + window_size < string.len() - 1 {
                // println!("Working");
                output.push(&string[n..n + window_size]);
                n = n + window_size;
            }
        }
        unique_values(&output)
    }

    pub fn split_alpha_numericals(string: String) -> (String, String) {
        /*
        "Something 123 else" => ("123","Something  else")
        */
        let bytes: Vec<_> = string.as_bytes().to_vec();
        let numbers: Vec<_> = bytes.iter().filter(|a| **a < 58 && **a > 47).collect();
        println!("{:?}", bytes);
        let aplhabets: Vec<_> = bytes
            .iter()
            .filter(|a| {
                (**a > 64 && **a < 91) // A-Z
                    || (**a > 96 && **a < 123) // a-z
                    || (**a > 127 && **a < 201) // letters with accents
                    || (**a == 32) // spaces
            })
            .collect();

        (
            // to have output as concatenated string
            String::from_utf8(numbers.iter().map(|a| **a).collect()).unwrap(),
            String::from_utf8(aplhabets.iter().map(|a| **a).collect()).unwrap(),
        )
    }

    pub fn char_count(string: String) -> BTreeMap<char, u32> {
        /*
        "SOmething Else" => {' ': 1, 'e': 3, 'g': 1, 'h': 1, 'i': 1, 'l': 1, 'm': 1, 'n': 1, 'o': 1, 's': 2, 't': 1}
         */
        let mut count: BTreeMap<char, Vec<i32>> = BTreeMap::new();
        let vector: Vec<_> = string.to_lowercase().chars().collect();

        // empty dictiornaty
        for i in vector.iter() {
            count.insert(*i, vec![]);
        }
        // dictionary with 1
        let mut new_count: BTreeMap<char, Vec<i32>> = BTreeMap::new();
        for (k, _) in count.iter() {
            let mut values = vec![];
            for i in vector.iter() {
                if i == k {
                    values.push(1);
                }
            }
            new_count.insert(*k, values);
        }

        // dictionary with sum of 1s
        let mut output = BTreeMap::new();
        for (k, v) in new_count.iter() {
            output.insert(*k, v.iter().fold(0, |a, b| a as u32 + *b as u32));
        }

        output
    }

    pub fn frequent_char(string: String) -> char {
        /*
            "SOmething Else" => 'e'
        */
        let dict = StringToMatch::char_count(string);
        let mut value = 0;
        let mut key = '-';
        for (k, v) in dict.iter() {
            key = match dict.get_key_value(k) {
                Some((x, y)) => {
                    if *y > value {
                        value = *y;
                        *x
                    } else {
                        key
                    }
                }
                _ => panic!("Please check the input!!"),
            };
        }
        key
    }

    pub fn char_replace(string: String, find: char, replace: String, operation: &str) -> String {
        /*
        ALL : SOmething Else is now "SOmZthing ElsZ"
        First : SOmething Else is now "SOmZthing Else"
        */

        if string.contains(find) {
            let string_utf8 = string.as_bytes().to_vec();
            let find_utf8 = find.to_string().as_bytes().to_vec();
            let replace_utf8 = replace.as_bytes().to_vec();
            let split = split_vector_at(&string_utf8, find_utf8[0]);
            let split_vec: Vec<_> = split
                .iter()
                .map(|a| String::from_utf8(a.to_vec()).unwrap())
                .collect();
            let mut new_string_vec = vec![];
            if operation == "all" {
                for (n, _) in split_vec.iter().enumerate() {
                    if n > 0 {
                        let x = split_vec[n][1..].to_string();
                        new_string_vec.push(format!(
                            "{}{}",
                            String::from_utf8(replace_utf8.clone()).unwrap(),
                            x.clone()
                        ));
                    } else {
                        new_string_vec.push(split_vec[n].clone());
                    }
                }
            } else {
                if operation == "first" {
                    for (n, _) in split_vec.iter().enumerate() {
                        if n == 1 {
                            let x = split_vec[n][1..].to_string();

                            new_string_vec.push(format!(
                                "{}{}",
                                String::from_utf8(replace_utf8.clone()).unwrap(),
                                x.clone()
                            ));
                        } else {
                            new_string_vec.push(split_vec[n].clone());
                        }
                    }
                } else {
                    panic!("Either pass operation as `all` or `first`");
                }
            }
            new_string_vec.concat()
        } else {
            panic!("The character to replace does not exist in the string passed, please check!")
        }
    }
}
