//

/*
Testing cmd parameters
*/

pub fn function() {
    // cargo test converts test code to test binary and runs it
    // By default all tests run in || and output is captured
    // since it is parallel, dependencies are not encourage between tests and no ordering either
    // cargo test goes to command line
    // cargo test -- goes to binary
    // cargo-test -- --test-thread=1 makes tests 1/1 and not ||ly
    // cargo-test -- --nocapture to show the printed values for success case
    // cargo-test <name_of_test> allows to run only this test, gives filtered details for the untested
    // to run multiple but not all test functions, the names should contain similarities ex:
    // cargo-test add will run add_two and add_three
    // same can be done by runnning a module and the once outside will not run
    // #[ignore] added before fn and after #[test] to ignore running that particular test
    // same in binary cargo test -- --ignored
}

#[test]
#[ignore]
fn huge_test_fucntion_to_be_avoided() {
    assert_eq!(1, 1);
}

#[test]
fn small_test_fucntion_to_be_run() {
    assert_eq!(1, 1);
}

/*
OUTPUT
running 2 tests
test _44_control_testing::huge_test_fucntion_to_be_avoided ... ignored
test _44_control_testing::small_test_fucntion_to_be_run ... ok

test result: ok. 1 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out
*/
