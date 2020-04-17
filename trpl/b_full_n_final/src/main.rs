use rand::Rng;
use std::cmp::Ordering;
use std::io;

fn main() {
    // making it play till they win or enter a non number
    loop {
        let random_nunber = rand::thread_rng().gen_range(1, 11); // generating a number
        println!("Enter a number between 1-10:",);
        // getting user input
        let mut user_entry = String::new();
        io::stdin()
            .read_line(&mut user_entry)
            .expect("Did not understand you!");
        // converting it to a unsigned 32
        let user_entry: u32 = match user_entry.trim().parse() {
            Ok(num) => num,
            Err(_) => break,
        };
        println!("> You have entered {}", user_entry);

        // comparing the entry with random number
        match user_entry.cmp(&random_nunber) {
            Ordering::Less => println!(
                "> Your entry is small for {}! Try again! or type Escape to quit",
                random_nunber
            ),
            Ordering::Greater => println!(
                "> Your entry is big for {}! Try again! or type Escape to quit",
                random_nunber
            ),
            Ordering::Equal => {
                println!("Thats a MATCH!",);
                break;
            }
        }
        println!();
    }
}

/*
OUTPUT
Enter a number between 1-10:
4
> You have entered 4
> Your entry is small for 8! Try again! or type Escape to quit

Enter a number between 1-10:
7
> You have entered 7
Thats a MATCH!
*/
