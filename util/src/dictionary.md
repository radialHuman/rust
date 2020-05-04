|# | Keyword/Function | From | Explanation |
|-|-|-|-|
|1. |use std::io::ErrorKind | trpl2 33 | Type of error using err.kind(), other_kind|
|2. |unwrap_or_else(\|x\| {}) | trpl2  33 | used to handle error and reduce nested match expressions|
|3. |unwrap()| trpl2 34 | to do match and panic in a shorter way|
|4. |except("")| trpl2 34 | to do unwrap() in a searchable way with unique message|
|5. |read_to_string() | trpl2 34 | to read contents of a read file, after File::open()|
|6. | ? | trpl2 34 | after an actions to check if it was scuccessful or will it propagate error, also changes error  type as per functions return type |
|7. | dyn Error | trpl2 34 | from use std::error::Error ??? |
|8. | panic!("") | trpl2 35 | error handling |
|9. | RUST_BACKTRACING=1  cargo run| trpl2 35| to show in cmd where the error ocurred and other details |
|10. | fn longest_string_with_lifetime_annotation<'a>(string1: &'a str, string2: &'a str) -> &'a str { | trpl2 41 | lifetime annotation (better in Doug Milford)|
|11. | assert!() | trpl2 42 | to check if something is true while testing |
|12. | #[should_panic(expected = "")] | trpl2 42 | to check and notify panic while testing |
|13. |cargo test | trpl2 44 | goes to command line|
|14. |cargo test --  | trpl2 44 |goes to binary|
|15. |cargo-test -- --test-thread=1  | trpl2 44 |makes tests 1/1 and not ||ly|
|16. |cargo-test -- --nocapture  | trpl2 44 |to show the printed values for success case|
|17. | cargo-test *name_of_the_test_function* | trpl2 44 | to test just that specific test function|
|18. | #[ignore] | trpl2 44 | added before fn and after #[test] to ignore running that particular test|