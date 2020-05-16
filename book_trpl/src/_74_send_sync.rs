// 16.4

/*
*/

pub fn function() {
    // std::marker has sync and send trait for concurrancy

    // SEND
    /*
    > ALl Rust types have send except Rc and Raw pointers
    > If Rc is cloned then counter gets updated twice
    > So Arc had send trait and can be sued for multiple thread
    */
    // SYNC
    /*
    > Any type is Sync if reference is Send
    */

    // Never manually use these traits as it is unsafe
    // More concurrancy are in crates as std lib is limited in terms of functionality
}

/*
OUTPUT
*/
