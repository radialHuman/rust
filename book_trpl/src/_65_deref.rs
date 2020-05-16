//

/*
* and Deref trait
*/

use std::ops::Deref; // to use dereferencing trait in CustomBox

pub fn function() {
    /*
    DEREFERENCE
    > Used to point to value  of a reference variable
    */
    let a = 10.;
    let b = &a;
    let c = Box::new(a);
    // assert_eq!(10., b); // this will not work as b is just a reference
    assert_eq!(10., *b); // while *b is actually the value
    assert_eq!(10., *c); // Box can be used instead of * as box has the dref trait in it

    // creating custom box type to understand its behaviour
    let a = 10.;
    let c = CustomBox::new(a);
    assert_eq!(10., *c); // will not work if Deref trait is not implemented in CustomBox like in Box
                         // this actually calls *(c.deref())

    // deref coersion converts &String to &str automatically else the code would have been &(*s)[..] instead of &s
    // deref coersion cant happen from a immutable reference to mutable reference because of borrowing rules
}

struct CustomBox<T>(T); // tuple struct
impl<T> CustomBox<T> {
    fn new(x: T) -> CustomBox<T> {
        CustomBox(x)
    }
}
impl<T> Deref for CustomBox<T> {
    type Target = T; // ??? associative types to declare generic parameter Chapter 19
    fn deref(&self) -> &T {
        &self.0 // access the first element of tuple from the struct
    }
}
/*
OUTPUT
*/
