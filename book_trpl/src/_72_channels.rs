//

/*
Tx and Rx values from single and multiple source
*/
use std::sync::mpsc;
use std::sync::mpsc::channel;
use std::thread;
use std::time::Duration;

pub fn function() {
    /*
    CHANNEL
    > Has rx and tx
    > If one is dropped the channel si closed
    >
    */
    let (tx, rx) = channel();

    // this spawn thread owns the tx now which has send to send a result type
    thread::spawn(move || {
        let val = "Something".to_string();
        println!("{}", val);
        tx.send(val).unwrap();
        // from here on val cant be used as it has been trasnfered to another thread
        // println!("{}", val);// this cant happen as the receiving thread is not the owner of val
    });

    let receiver = rx.recv().unwrap();
    // recv blocks main thread until the value is sent
    println!("{} received", receiver);

    // Sending multiple values to receiver
    let (tx, rx) = channel();

    thread::spawn(move || {
        let val = vec![3, 5, 7, 8, 9, 2];
        for i in val {
            tx.send(i).unwrap();
            thread::sleep(Duration::from_secs(1));
        }
    });
    // once the tx is over, it exists as tx drops

    // rx can be iterator and recv is implied
    for i in rx {
        println!("The value is {}", i);
    }

    // Multiple producers by cloning
    println!("\n\nFrom multiple txs:");
    let (tx, rx) = channel();
    let tx1 = mpsc::Sender::clone(&tx);
    // first tx
    thread::spawn(move || {
        let val = vec![0; 6];
        for i in val {
            tx1.send(i).unwrap();
            thread::sleep(Duration::from_secs(1));
        }
    });
    // second thread
    thread::spawn(move || {
        let val = vec![3, 5, 7, 8, 9, 2];
        for i in val {
            tx.send(i).unwrap();
            thread::sleep(Duration::from_secs(1));
        }
    });

    for i in rx {
        println!("The value is {}", i);
    }
}

/*
OUTPUT
Something
Something received
The value is 3
The value is 5
The value is 7
The value is 8
The value is 9
The value is 2


From multiple txs:
The value is 0
The value is 3
The value is 0
The value is 5
The value is 7
The value is 0
The value is 0
The value is 8
The value is 0
The value is 9
The value is 2
The value is 0
*/
