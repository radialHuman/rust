// 15.4

/*
How to have multiple references using reference counter smart pointer Rc
*/

pub fn function() {
    // RC (Reference counting)
    /*
    > If a value has multiple owners, all the owners show disappear before destroying the value
    > Keeps track of a value and how many owners it has
    > USed when its not known whose will be the last owner
    > Usable only in single threaded
    > Exmaple here has two seperate list that merge into another list
    */
    // let a = Cons(5, Box::new(Cons(10,Box::new(Nil))));
    // let b =  Cons(3, Box::new(a));
    // let c =  Cons(4, Box::new(a)); // would not work as a is now moved to b
    /*
    > This can be avoided by using &
    > Which would need lifetime specification
    > Which will lead to a erro with NIL ???
    */

    // so Rc is the alternative, which will clone the pointer to the data on the heap
    let a = Rc::new(Cons(5, Rc::new(Cons(10, Rc::new(Nil)))));
    println!("After a, count of Rc of a is  {}", Rc::strong_count(&a));
    let b = Cons(3, Rc::clone(&a)); // better than .clone() as no deep copy is involved and incrementing the count doesn take much time
    println!("After b, count of Rc of a is  {}", Rc::strong_count(&a));
    {
        let c = Cons(4, Rc::clone(&a));
        println!(
            "After c in inner scope, count of Rc of a is  {}",
            Rc::strong_count(&a)
        );
    } // the count decreases as the drop is implemented at the end of scope
    println!(
        "After inner scope, count of Rc of a is  {}",
        Rc::strong_count(&a)
    );

    // multiple mutalbe references are avoided as it can lead to data races or inconsistencies
}

enum List {
    // Cons(i32, Box<List>), // replaced with
    Cons(i32, Rc<List>),
    Nil,
}
use std::rc::Rc;
use List::{Cons, Nil};

/*
OUTPUT
After a, count of Rc of a is  1
After b, count of Rc of a is  2
After c in inner scope, count of Rc of a is  3
After c in main scope, count of Rc of a is  2
*/
