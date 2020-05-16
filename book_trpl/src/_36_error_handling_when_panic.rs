// 9.3

/*
When to use panic and result
*/

pub fn function() {
    // PANIC vs RESULT
    /*
    When panic, it makes error unrecoverable no matter what (for examples, testing and prototyping to understand what sort of errors can occur)
    Result instead can maintain both (usually a better option)
    unwrap and except can also be used when its certain that it will not be an error
    */
    let s: i32 = "32".parse().unwrap();
    // even though 32 is a perfect string still parse gives result and so unwrap can be used
    // The same thing if sent by a user, can be incorrect and then Result can be used deu to uncertaininty of the situation
    println!("{}", s);
}
