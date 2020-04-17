fn main() {
    println!("\n**** IF ****");
    let num = 137;

    if num % 2 == 0 {
        println!("{} is divisible by 2", num);
    } else if num % 3 == 0 {
        println!("{} is divisible by 3", num);
    } else if num % 5 == 0 {
        println!("{} is divisible by 5", num);
    } else if num % 7 == 0 {
        println!("{} is divisible by 7", num);
    } else if num % 11 == 0 {
        println!("{} is divisible by 11", num);
    } else if num % 13 == 0 {
        println!("{} is divisible by 13", num);
    } else {
        println!("{} is likely to be a prime", num);
    }

    println!("\n**** IF IN LET ****");
    let bol = true;
    let mut num = if bol {
        1
    } else {
        0 // has to be of same data type as the return in if statment
    };
    println!("The number from let in if is {}", num);

    println!("\n**** LOOP ****");
    loop {
        if num < 5 {
            println!("Loop - {}", num);
        } else {
            break;
        }
        num += 1;
    }
    // returning value form loop, can be donw by assigning loop to a variable
    // WHILE LOOP, cleaner loop, avoids break
    while num != 10 {
        println!("While - {}", num);
        num += 1;
    }
    // FOR LOOP better in case of collections, safer
    let arr = [45, 23, 78, 21, 79, 21];
    for i in arr.iter() {
        println!("For - {}", i);
    }

    for _ in (10..=20).rev() {
        // range based loop
        print!("For Loop | ");
    }
}
/*
OUTPUT
**** IF ****
137 is likely to be a prime

**** IF IN LET ****
The number from let in if is 1

**** LOOP ****
Loop - 1
Loop - 2
Loop - 3
Loop - 4
While - 5
While - 6
While - 7
While - 8
While - 9
For - 45
For - 23
For - 78
For - 21
For - 79
For - 21
For Loop | For Loop | For Loop | For Loop | For Loop | For Loop | For Loop | For Loop | For Loop | For Loop | For Loop |
*/
