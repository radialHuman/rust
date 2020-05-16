// 15.3

/*
drop trait vs std::mem::drop
*/

struct CustomeString {
    data: String,
}

impl Drop for CustomeString {
    fn drop(&mut self) {
        println!("Dropping the memeory of {}", &self.data); // replaced with clean up code in general
    }
}

pub fn function() {
    // DROP trait
    /*
    > Another trait implemented in Box<T>
    > used for cleaning up heap memory
    > dropping of variables occurs in reverse order of assignment within the scope
    > To understand how drop works
    */
    let a = CustomeString {
        data: "First".to_string(), // drop is called automatically
    };
    {
        // this scope comes to an end before the other so this will drop first then println is executed and then a is dropped
        let b = CustomeString {
            data: "Second".to_string(), // drop is called automatically
        };
    }
    let c = CustomeString {
        data: "Third".to_string(), // drop is called automatically
    };
    println!("ALl clear",);

    // FORCED DROP
    /*
    > To drop something forcefully std::mem::drop
    > drop trait from struct/enum cant be called in this case
    > as that would run again when the scope ends creating confusion (double free error)
    */
    // drop(c); // this is std::mem::dropunlike c.drop();
}

/*
OUTPUT
Dropping the memeory of Second
ALl clear
Dropping the memeory of Third
Dropping the memeory of First
*/
