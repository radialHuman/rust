// 15.5

/*
*/

pub fn function() {
    // Interior mutability is breaking all the rules and mutating immutable variables using UNSAFE
    // This can be wrapped in SAFE API and considered as immutable
    // RefCell enforces rules at runtime and panics while Box applies ruels at compile time
    // If done in compile time (majority) it does not affect performance
    // While in Runtime allows certain things which are not in previous case. Few analysis are impossible due to complie time being conservative
    // Halting problem ???
    // Rust can avoid running few programs if it does not think it is safe even if it is that way it builds trust with the user
    // It allows false negative but not false positive
    // Used in single threaded case
    // Box and RefCell single owner of data
    // Box and RefCell allow mutable or immutable borrows at compile and runtime respectively
    // An immutable variable cant have mutable reference

    let x = 10;
    // let y = &mut x; // This cant happen as x is mutable

    // If a variable is supposed to be immutable to everyone but mutate in some functions it can be done using RefCell
    // Borrowing rules are checked in runtime
    use std::cell::RefCell;
    // vec![] becomes RefCell::new(vec![]) and
    // values are passed as .borrow_mut()/.borrow()

    // Rc and RefCell can be combined to mutate multiple 
}

/*
OUTPUT
*/
