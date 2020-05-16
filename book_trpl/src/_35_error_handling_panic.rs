// 9.1 PANIC!

/*
How panic works and whats a backtrace
*/

pub fn function() {
    /* There is no exception in Rust
    It treats errors as
    > Recoverable (bad input) -> Result<T, E>
    > Unrecoverable (bug kind) -> panic!("...")
    once a error is detected, either recover from it or stop execution
    */

    // PANIC
    /*
    When programmer is not sure how to handle the error
    This shows the error message, unwinds and cleans up the stack before quitting
    This requires time and space as it has to clean up, instead an alternate would be to use abort
    using
    [profile.release]
    panic = 'abort'
    skips the part of cleaning and just aborts and lts the os do the cleaning
    keeps the binary file smaller
    */
    panic!("This were it went downhill");

    /*
    BACKTRACE
    Using this: (if debug is enabled for caro run and build)
    RUST_BACKTRACE=1 cargo run
    shows the function where the panic occurs if it is in someother function
    and gives the details
    Read till the file executed is visible
    */
}
