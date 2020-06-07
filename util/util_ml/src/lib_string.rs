/*
DESCRIPTION
-----------------------------------------
STRUCTS
-------
1. StringToMatch :
    > compare_percentage
        x calculate
    > clean_string
        x char_vector
    > compare_chars
    > compare_position

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
}
