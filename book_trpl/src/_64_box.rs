//
/*
Box intro
*/

pub fn function() {
    /*
    SMART POINTERS (SP)
    > &, a pointer,  borrows data they point to and nothing else
    > SP are data structures that point to something and unlike & has other capabilities like owning it too etc
    > String, Vec<T> are a type of SP
    > SP are implemented using structs and have traits like Deref and Drop
    > There are many in std lib, covered here are:
        > Box<T> : Allocates value on heap and pointer on stack
        > Rc<T> : refernce counting type that enables multiple ownership
        > Ref<T> :
        > RefMut<T> :
        > RefCell<T> : Borrows rule at run time than during complie time
    */

    /*
    BOX<T>
    > When a types size is unknown at compile time and its needed to be used where extact type is required to be known
    > When large data ownership is to be transfered but not copied
    > When a particular values' tyep is required due to its traits ??? in Chapter 17
    */
    let b = Box::new(5.);
    println!("Box has {} in it.", b);
    // when the scope ends the pointer on stack and data on heap is deallocated
    // i32 is a small one and goes to stack by default, to use box the scenario is different like unknown size during complie time
    // one such type is recursive type

    // making a cons list (list with last element Nil) using enum
    // this wont compile as size of enum is unknown
    // Using box it uses pointer on stack whose size is known while the data remains in heap
    let l1 = List::Cons(
        1,
        Box::new(List::Cons(2, Box::new(List::Cons(3, Box::new(List::Nil))))),
    );
    // to avoid List::Cons and List::Nil, import it using use List::{Cons,Nil};

    // There is no other over head as it has no other spl feature apart form indirection like other smart pointers
    // The two traits of Box:
    // > Deref to treat values of box as reference
    // > Drop to clear values from heap once out of scope
}

// this would not work
// enum List {
//     Cons(i32, List),
//     Nil,
// }

// to make it work it has to be boxed
enum List {
    Cons(i32, Box<List>),
    Nil,
}

/*
OUTPUT
b = 5
*/
