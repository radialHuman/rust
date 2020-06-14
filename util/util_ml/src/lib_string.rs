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
1. extract_vowels_consonants : Returns a tuple of vectors containing chars (after converting to lowercase)
    > 1. string : String
    = Vec<chars> 
    = Vec<chars>

2. sentence_case
    > 1. string : String
     = String

3. remove_stop_words : Based on NLTK removing words that dont convey much from a string
    > 1. string : String
     = String

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


pub fn extract_vowels_consonants(string: String) -> (Vec<char>, Vec<char>) {
    /*
    Returns a tuple of vectors containing chars (after converting to lowercase)
    .0 : list of vowels
    .1 : list fo consonants
    */
    let bytes: Vec<_> = string.as_bytes().to_vec();
    let vowels: Vec<_> = bytes
        .iter()
        .filter(|a| {
            **a == 97
                || **a == 101
                || **a == 105
                || **a == 111
                || **a == 117
                || **a == 65
                || **a == 69
                || **a == 73
                || **a == 79
                || **a == 85
        })
        .collect();
    let consonants: Vec<_> = bytes
        .iter()
        .filter(|a| {
            **a != 97
                && **a != 101
                && **a != 105
                && **a != 111
                && **a != 117
                && **a != 65
                && **a != 69
                && **a != 73
                && **a != 79
                && **a != 85
                && ((**a > 96 && **a < 123) || (**a > 64 && **a < 91))
        })
        .collect();
    let output: (Vec<_>, Vec<_>) = (
        String::from_utf8(vowels.iter().map(|a| **a).collect())
            .unwrap()
            .chars()
            .collect(),
        String::from_utf8(consonants.iter().map(|a| **a).collect())
            .unwrap()
            .chars()
            .collect(),
    );
    output
}

pub fn sentence_case(string: String) -> String {
    /*
    "The quick brown dog jumps Over the lazy fox" => "The Quick Brown Dog Jumps Over The Lazy Fox"
    */
    let lower = string.to_lowercase();
    let split: Vec<_> = lower.split(' ').collect();
    let mut output = vec![];
    for i in split.iter() {
        let char_vec: Vec<_> = i.chars().collect();
        let mut b = [0; 2];
        char_vec[0].encode_utf8(&mut b);
        output.push(format!(
            "{}{}",
            &String::from_utf8(vec![b[0] - 32 as u8]).unwrap()[..],
            &i[1..]
        ));
    }
    output.join(" ")
}

pub fn remove_stop_words(string: String) -> String {
    /*
    "Rust is a multi-paradigm programming language focused on performance and safety, especially safe concurrency.[15][16] Rust is syntactically similar to C++,[17] but provides memory safety without using garbage collection.\nRust was originally designed by Graydon Hoare at Mozilla Research, with contributions from Dave Herman, Brendan Eich, and others.[18][19] The designers refined the language while writing the Servo layout or browser engine,[20] and the Rust compiler. The compiler is free and open-source software dual-licensed under the MIT License and Apache License 2.0."
                                                                                                    |
                                                                                                    V
    "Rust multi-paradigm programming language focused performance safety, especially safe concurrency.[15][16] Rust syntactically similar C++,[17] provides memory safety without using garbage collection.\nRust originally designed Graydon Hoare Mozilla Research, contributions Dave Herman, Brendan Eich, others.[18][19] designers refined language writing Servo layout browser engine,[20] Rust compiler. compiler free open-source software dual-licensed MIT License Apache License 2.0."
         */
    // https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip (14/06/2020)
    let mut split: Vec<_> = string.split(' ').collect();
    let stop_words = vec![
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "you're",
        "you've",
        "you'll",
        "you'd",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "she's",
        "her",
        "hers",
        "herself",
        "it",
        "it's",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "that'll",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "don't",
        "should",
        "should've",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "aren't",
        "couldn",
        "couldn't",
        "didn",
        "didn't",
        "doesn",
        "doesn't",
        "hadn",
        "hadn't",
        "hasn",
        "hasn't",
        "haven",
        "haven't",
        "isn",
        "isn't",
        "ma",
        "mightn",
        "mightn't",
        "mustn",
        "mustn't",
        "needn",
        "needn't",
        "shan",
        "shan't",
        "shouldn",
        "shouldn't",
        "wasn",
        "wasn't",
        "weren",
        "weren't",
        "won",
        "won't",
        "wouldn",
        "wouldn't",
        "I",
        "Me",
        "My",
        "Myself",
        "We",
        "Our",
        "Ours",
        "Ourselves",
        "You",
        "You're",
        "You've",
        "You'll",
        "You'd",
        "Your",
        "Yours",
        "Yourself",
        "Yourselves",
        "He",
        "Him",
        "His",
        "Himself",
        "She",
        "She's",
        "Her",
        "Hers",
        "Herself",
        "It",
        "It's",
        "Its",
        "Itself",
        "They",
        "Them",
        "Their",
        "Theirs",
        "Themselves",
        "What",
        "Which",
        "Who",
        "Whom",
        "This",
        "That",
        "That'll",
        "These",
        "Those",
        "Am",
        "Is",
        "Are",
        "Was",
        "Were",
        "Be",
        "Been",
        "Being",
        "Have",
        "Has",
        "Had",
        "Having",
        "Do",
        "Does",
        "Did",
        "Doing",
        "A",
        "An",
        "The",
        "And",
        "But",
        "If",
        "Or",
        "Because",
        "As",
        "Until",
        "While",
        "Of",
        "At",
        "By",
        "For",
        "With",
        "About",
        "Against",
        "Between",
        "Into",
        "Through",
        "During",
        "Before",
        "After",
        "Above",
        "Below",
        "To",
        "From",
        "Up",
        "Down",
        "In",
        "Out",
        "On",
        "Off",
        "Over",
        "Under",
        "Again",
        "Further",
        "Then",
        "Once",
        "Here",
        "There",
        "When",
        "Where",
        "Why",
        "How",
        "All",
        "Any",
        "Both",
        "Each",
        "Few",
        "More",
        "Most",
        "Other",
        "Some",
        "Such",
        "No",
        "Nor",
        "Not",
        "Only",
        "Own",
        "Same",
        "So",
        "Than",
        "Too",
        "Very",
        "S",
        "T",
        "Can",
        "Will",
        "Just",
        "Don",
        "Don't",
        "Should",
        "Should've",
        "Now",
        "D",
        "Ll",
        "M",
        "O",
        "Re",
        "Ve",
        "Y",
        "Ain",
        "Aren",
        "Aren't",
        "Couldn",
        "Couldn't",
        "Didn",
        "Didn't",
        "Doesn",
        "Doesn't",
        "Hadn",
        "Hadn't",
        "Hasn",
        "Hasn't",
        "Haven",
        "Haven't",
        "Isn",
        "Isn't",
        "Ma",
        "Mightn",
        "Mightn't",
        "Mustn",
        "Mustn't",
        "Needn",
        "Needn't",
        "Shan",
        "Shan't",
        "Shouldn",
        "Shouldn't",
        "Wasn",
        "Wasn't",
        "Weren",
        "Weren't",
        "Won",
        "Won't",
        "Wouldn",
        "Wouldn't",
    ];
    split.retain(|a| stop_words.contains(a) == false);
    split
        .iter()
        .map(|a| String::from(*a))
        .collect::<Vec<String>>()
        .join(" ")
}
