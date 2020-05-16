// 11 Testing

/*
Testing with asset, assert_eq and _ne, panic, should_panic and Result
these with custom messages shown

*/

pub fn function() {
    /*
    > testing sets up the state
    > Runs the code to be tested
    > Asserts the actual output to desired output
    > Tests run parallelly
     */
    println!("{} + 2 = {}", 20, add_two(20));
}

// to change a function to test fucntion add an attribute before the fn like so
#[test]
fn tester_add_two() {
    assert_eq!(7, add_two(5)); // left equals right
    assert_ne!(add_two(5), 8); // not equals, order does not matter in upper and in this line
}

pub fn add_two(i: i32) -> i32 {
    i + 2
}

// function to see if a rectangle can fit in another and testing it
struct Rectangle {
    width: u8,
    height: u8,
}
impl Rectangle {
    pub fn can_it_fit_in(&self, another_rectangle: &Rectangle) -> bool {
        self.width > another_rectangle.width && self.height > another_rectangle.height
    }
}

use super::*; // for accessing fucntions outside scope
#[test]
fn fit_rectangle_test() {
    let smaller = Rectangle {
        width: 8,
        height: 8,
    };
    let larger = Rectangle {
        width: 80,
        height: 9,
    };
    assert!(
        !larger.can_it_fit_in(&smaller),
        "It cant fit {}x{} vs {}x{} ",
        smaller.height,
        smaller.width,
        larger.height,
        larger.width
    );
}
// doing just to opposite
/*
> #[derive(PartialEq,Debug)] is added to function incase printing out is required
> error messgaes also can be added after passing values to assert...!
> And parameters passed on to it acts like format!
*/
#[test]
fn cant_fit_rectangle_test() {
    let smaller = Rectangle {
        width: 8,
        height: 8,
    };
    let larger = Rectangle {
        width: 80,
        height: 9,
    };
    assert!(!smaller.can_it_fit_in(&larger),);
}

fn panic_function() {
    panic!("This is panicking!!");
}
#[test]
#[should_panic(expected = "Expected it to panic")]
fn panic_tester() {
    panic_function();
}

// test using result
#[test]
fn add_two_to_result() -> Result<(), String> {
    if 2 + 2 == 5 {
        Ok(())
    } else {
        Err("Something is wrong".to_string())
    }
}

/*
OUTPUT (Only errors; code will not run)
running 2 tests
test _42_43_testing::cant_fit_rectangle_test ... thread 'ok_42_43_testing::fit_rectangle_test
' panicked at 'It cant fit 8x8 vs 9x80 ', src\_42_43_testing.rs:49:5

---- _42_43_testing::panic_tester stdout ----
note: panic did not contain expected string
      panic message: `"This is panicking!!"`,
 expected substring: `"Expected it to panic"`

 running 1 test
Error: "Something is wrong"
thread '_42_43_testing::add_two_to_result' panicked at 'assertion failed: `(left == right)`
  left: `1`,
 right: `0`:
*/
