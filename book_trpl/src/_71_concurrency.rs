// 16.1

/*
Whats concurrency, threads, how they work
block by join
move environement variable to thread
*/

use std::thread;
use std::time::Duration;

pub fn function() {
    /*
    > Coincidentally, solving borrow, owenrship and lifetime, solved the issues related to concurrency
    > Its the ability to run part of program independently (||ism is runnign them at the same time)
    > Rust takes care at compile time(good) than in run time(bad)
    > Avoids bugs and bugs due to refactoring
    > Rust beign low level, takes care of performance for these problems
    > Using threads
    */

    /*
     * Threads : An OS has many process running simultaneously. Similarly, within a program, threads run independent parts simultaneously
     * Due to this, it increases complexity, and there is no control over order of execution of these threads causing
     *  > Race condition : threads access data inconsistently
     *  > Deadlock : where two threads waits for each other hence not proceeding
     *  > Other corner cases which are difficult to fix
     */

    // Multithreaded needs different type of program structure and is complicated from single
    // There are ways to implement this like 1:1 call for thread or green thread model M:N where M threads using N os
    // Rust focuses on runtime which is the code implemented by Rust in every binary
    // Less runtime, less features in it, more compatibility with other language and small binary file
    // Rust aims at no runtime
    // M:N needs large runtime to manage threads so it uses 1:1
    // But there are crates to do M:N, gives more control over running which thread when and ability to reduce cost of context switch ???

    thread::spawn(|| {
        for i in 1..10 {
            println!("From spwan thread : {}", i);
            thread::sleep(Duration::from_secs(1));
        }
    });

    // As soon as the main thread ends, it doesn matter but the program stops without caring abut the spawned
    // if all spawend are to be executed use join on the variable carrying spawned
    let spawner = thread::spawn(|| {
        for i in 100..105 {
            println!("From joined spwan thread : {}", i);
            thread::sleep(Duration::from_secs(1));
        }
    });
    spawner.join().unwrap();
    // join blocks the thread so if it was before main thread, it will complete spawn before showing main thread

    for i in 10..15 {
        println!("From main thread : {}", i);
        thread::sleep(Duration::from_secs(1));
    }

    // MOVE
    /*
    // TO move variable between two thread
    // In this case if main has to pass on the vector to spawner, it has pt be moved
    // As the vector may not last till the spawner thread ends
    // Move makes spawner thread take ownership than borrow and later missing it
    /
     */
    let v = vec![3, 4, 1];
    thread::spawn(move || {
        println!("The vector is {:?}", v);
    })
    .join()
    .unwrap();
}

/*
OUTPUT
From spwan thread : 1
From joined spwan thread : 100
From joined spwan thread : 101
From spwan thread : 2
From joined spwan thread : 102
From spwan thread : 3
From joined spwan thread : 103
From spwan thread : 4
From joined spwan thread : 104
From spwan thread : 5
From main thread : 10
From spwan thread : 6
From main thread : 11
From spwan thread : 7
From main thread : 12
From spwan thread : 8
From main thread : 13
From spwan thread : 9
From main thread : 14
The vector is [3, 4, 1]
*/
