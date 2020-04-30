| Keyword/Function | From | Explanation |
|-|-|-|
|use std::io::ErrorKind | trpl2 | Type of error using err.kind(), other_kind|
|unwrap_or_else(\|x\| {}) | trpl2 | used to handle error and reduce nested match expressions|
|unwrap()| trpl2 | to do match and panic in a shorter way|
|except("")| trpl2 | to do unwrap() in a searchable way with unique message|
|read_to_string() | trpl2 | to read contents of a read file, after File::open()|
| ? | trpl2 | after an actions to check if it was scuccessful or will it propagate error, also changes error type as per functions return type |
| dyn Error | trpl2 | from use std::error::Error ??? |
