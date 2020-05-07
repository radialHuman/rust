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
|19. | collect | trpl2 45 | to make a vector of somthing, needs type specification |
|20. | clone() | trpl2 45 | to make a copy of data using speed and memory, can be avoided |
|21.| use std::fs::File;| Trpl2 33 | To open a file from local|
|21.a| fs::read_to_string| Trpl2 33| To read a file intoa string|
|22. | use std::io::Read;| Trpl2 34 | To open and read a file from local|
|23. | use super::*|??? | ???|
|24. | use std::env;|Trpl2 45 | To capture user input passed as parameter|
|25. | use std::thread; |Trpl2 52| ??? |
|26. | use std::time::Duration;| Trpl2 53| sleep |
|27. | move|???|Uses copy trait|
|28. | Fn| Trpl2 52| ???|
|29. | FnOnce| Trpl2 52| ???|
|30. | FnMut| Trpl2 52| ???|
|31. |iter()| Trple 55| To traverse through collections with immutable reference|
|32. |iter_mut()|Trple 55| To traverse through collections with mutable reference|
|33. |into_iter()|Trple 55| To traverse through collections by taking ownership|
|34. |.filter() |Trple 55 | To filter a collection|
|35. |.map() |Trple 55 | To modify all elements a collection|
|36. |.sum()|Trple 55 | To add up a collection|
|37. |.collect()|Trple 55 | To finish lazily called functions like filter and map|
|38. |.zip()|Trple 55 | To make tuples of two collections |
|39. | >> | ??? | Bit shift|
|40. |"Unrolling"| Trpl2 55 | Used to optimize for loops but iterating |