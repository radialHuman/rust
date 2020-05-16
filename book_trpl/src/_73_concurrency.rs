// 16.3

/*
Mutex for multiple thread sharing a value 1/1
Rc are not thread safe so
Arc for concurrant Rc
*/

use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::thread;

pub fn function() {
    /*
    > Message passing is not the only way to do concurrent coding
    > Channel is single ownership type
    > While Shared memory concurrency is multiple threads accessing a single data
    > Mutex is sued in this case, where a thread
        > Asks for access
        > Uses data
        > Frees up the access for others to use
    */
    /* MUTEX : Mutual exclusion
    > Asks for accesss to data (mutex lock)
    > Rust's borrowing and ownership principle makes it easy to handle unlike other languages where it becomes complicated
    > Mutex has interior mutability
    */

    // in sigle threaded context
    let m = Mutex::new(5);
    {
        let mut num = m.lock().unwrap(); // to block the current thread using lock
                                         // lock() brings in SP called MutexGuard for Deref-ing inner data which calls for LockResult
        *num = 6;
    } // this drops the lock
    println!("The values is {:?}", m);

    // in multiple threaded context, use of Rc due to multiple ownership
    let counter = Arc::new(Mutex::new(5)); // creating a mutex with multiple possible owners, It was Rc
    let mut vector = vec![]; // to append changed data

    for _ in 0..10 {
        let counter = Arc::clone(&counter); // this was Rc, with error
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        vector.push(handle);
    }
    for i in vector {
        i.join().unwrap(); // to block the thread and move on
    }

    println!("Result : {}", *counter.lock().unwrap());
}

/*
OUTPUT
The values is Mutex { data: 6 }
Result : 15
*/
