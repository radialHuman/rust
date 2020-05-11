|# | Keyword/Function | From | Explanation |
|-|-|-|-|
|1. | use std::io::ErrorKind | trpl2 33 | Type of error using err.kind(), other_kind|
|2. | unwrap_or_else(\|x\| {}) | trpl2  33 | used to handle error and reduce nested match expressions|
|3. | unwrap()| trpl2 34 | to do match and panic in a shorter way|
|4. | except("")| trpl2 34 | to do unwrap() in a searchable way with unique message|
|5. | read_to_string() | trpl2 34 | to read contents of a read file, after File::open()|
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
|31. | iter()| Trple 55| To traverse through collections with immutable reference|
|32. | iter_mut()|Trple 55| To traverse through collections with mutable reference|
|33. | into_iter()|Trple 55| To traverse through collections by taking ownership|
|34. | .filter() |Trple 55 | To filter a collection|
|35. | .map() |Trple 55 | To modify all elements a collection|
|36. | .sum()|Trple 55 | To add up a collection|
|37. | .collect()|Trple 55 | To finish lazily called functions like filter and map|
|38. | .zip()|Trple 55 | To make tuples of two collections |
|39. | >> | ??? | Bit shift|
|40. | "Unrolling"| Trpl2 55 | Used to optimize for loops but iterating |
|41. | Box<T>| Trpl2 64| Smart pointer to store value on heap and pointer on stack|
|42. | Deref coersion| Trpl2 65| Uses automatic conversion of type ex: &String to &str|
|43. | Associative types |???|???|
|44. | Double free error| Trpl2 66| When destructor like drop is called and drop of the same value occurs when it goes out of scope |
|45. | std::rc::Rc| Trpl2 67| Smart Pointer (SP) To have multiple reference |
|46. | Rc::clone(& *variable*)| Trpl2 67| Reference counting of a variable, cloning pointer referencing to heap memory |
|47. | Rc::strong_count(& *variable*)| Trpl2 67| Shows the current # of references to the variable |
|48. | RefCell<T>| Trpl2 68|use std::cell::RefCell; to mutate immutable objects (unsafe)|
|48. | Cell<T>| Trpl2 68|use std::cell::RefCell; to get data in and out (unsafe)|
|49. | .borrow() |Trpl2 68| Used instead of & while using RefCell; has SP Ref<T> and Deref|
|50. | .borrow_mut() |Trpl2 68| Used instead of &mut while using RefCell; has SP RefMut<T> and Deref|
|51. | thread::spawn(||)| Trpl2 71| Thread related API |
|52. | thread::sleep(Duration::from_secs())| Trpl2 71| To sleep|
|53. | spawner.join().unwrap();| Trpl2 71| To block the thread and move on once its over|
|54. | thread::spawn(move \|\| {})| Trpl2 71| TO use environment variable in thread, it has to tkae ownership and cant borro as its not sure when the variable would get dropped|
|55. | std::sync::mpsc::channel;| Trpl2 72| TO create channel *multi producer single consumer*|
|56. | tx.send(value).unwrap();| Trpl2 72| To send value via transmitter form a thread|
|57. | rx.recv().unwrap();| Trpl2 72| To receive value sent via tx; blocks main thread until the value is sent |
|58. | rx.try_recv().unwrap();| Trpl2 72| To receive value sent via tx; does not block while the main thread can do soem other operations|
|59. | tx1 = mpsc::Sender::clone(&tx)| Trpl2 72| To clone a tx|
|60. | use std::sync::Mutex; | Trpl2 73| Mutex for shared data concurrency |
|61. | std::sync::Arc; | Trpl2 73| Smart Pointer (SP) Atomic reference counter To have multiple reference for mutex |
|62. | Send | Trpl2 74| Trait from std::marker for concurrancy sending|
|63. | Sync | Trpl2 74| Trait from std::marker for concurrancy syncing|