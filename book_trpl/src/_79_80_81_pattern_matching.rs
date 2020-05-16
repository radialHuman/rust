// 18.1/2

/*
Pattern usage in various places
Refutable n irrefutable patterns
Destucturing
Ingmoring
Binding
Shadow variable
Match guard
*/

pub fn function() {
    //MATCH ARM
    /*
    > Needs to cover all patterns
    > If not then end it with _ => Expression, which ignores any other pattern
    */

    // If let
    /*
    > Is short way of match for one case
    > This can have else condition too
    > Cant have a conditionals in it, has to be inside another scope
    > Is not as exhaustive as match
    */

    // While let
    /*
    > Runs till the statement is not true
    */
    let mut list = vec![6, 5, 4, 3, 2, 1];
    while let Some(value_in_list) = list.pop() {
        println!("{}", value_in_list);
    }

    // For
    /*
    > Pattern in forloop is the variable after it and can be destructured ex: enumerate
    > for (i,j) in list.iter().enumerate(){...}
    */

    // Let
    /*
    > Similarly for let (a,b,_) = (1,2,3);
    > a gets 1 b gets 2 and 3 is ignored
    */

    // Functions
    /*
    > Similarly fn some_function(&(x,y): &(i32, u8)){...}
    */

    // Irrefutable and refutable patterns
    /*
    > For, let uses irref.
    > If let and while let uses both
    */
    // let x = some_value; // will not work as it can be none and will show
    // if let Some(x) = some_value {...} // this can be used instead

    // Shadow variable
    let x = Some(5);
    let y = 10;
    let z = 'r';

    match x {
        Some(1..=10) => println!("Less than 10"),
        Some(50) => println!("Its 50"),
        Some(y) => println!("Its {}", y), // this is the shadow variable which makes inner y =5 not 10 while the outer remains untouched
        _ => println!("X is {:?}", x),
    };

    println!("Originals as {:?}, {}", x, y); // this will show outer x and y not the shadow as it got dropped

    // to reference y inside the scope and not the shadow variable, match guard is used

    match x {
        Some(1..=10) => println!("Less than 10"),
        Some(50) => println!("Its 50"),
        Some(n) if n == y => println!("Its {}", y), // match guard is a confition after pattern in match
        _ => println!("X is {:?}", x),
    };

    match y {
        1 | 5 => println!("Its one or five"), // multiple patterns
        _ => (),                              // do nothign
    }
    // match guard can be used for multiple matching conditions
    let w = true;

    match y {
        1 | 5 if w => println!("Its one or five"), // multiple patterns, like so (1|5) and true
        _ => (),                                   // do nothign
    }

    match z {
        'a'..='e' => println!("Its a character"), // range matching
        _ => (),
    }

    // DESRUCTURING
    // Can happen in let, enum, struct, function and nested of such structures

    // IGNORING _ and ..
    /*
    > let m = (1,2);
    > match m{
        (1,_) => {}
        _ => {}
    }

    > let m = (1,2,3,4,5);
    > match m {
        (first , .., last) => {} // just one .. is enough more than that can cause error
        (.., second ,..)// this will not work as it is not clear is can be (1,2,seond, ..) or (1, second, ..)
    }
    */

    // BINDING
    let msg = Message::Hello { id: 32 };
    match msg {
        Message::Hello {
            id: binding_variable @ 1..=50,
        } => println!("The valeus is between 1 and 50 {}", binding_variable),
        _ => (),
    }
    // println!("{}", binding_variable);// cant be accessed as it gets dropped after the scope
}

enum Message {
    Hello { id: i32 },
}

/*
OUTPUT
1
2
3
4
5
6
Less than 10
Originals as Some(5), 10
Less than 10
The valeus is between 1 and 50 32
*/
