pub fn code() {
    // core of RUST
    // new concept for all users
    // Its all about memory management
    // No garbage collection here
    // Due to this concept , the runtime is as fast as cpp
    // parallel and concurrent is possible cos of this easily
    // safety is asssured
    println!("**** OWNERSHIP ****",);
    println!("STACK AND HEAP",);
    println!(
        "Stack:
> Is for fast information storage and retrival
> Once the variable goes out of scope it will recapture the memory
> It is default in RUST
> It store values that have predefined size like int float bool char but not strings
> Stack is LIFO
> Fast because everything is stacked in order and pointer can easily find it
> Garbage collection happens manually by defining the scopes
\n\n\n
Heap:
> Variables in heap can grow in size 
> Vectors, strings , hashmaps can exist here
> Runtime is affected due to this dynamic nature
> Memory can live beyond the scope of creation
> Memory is recaptured, when the LAST OWNER goes out of scope
    ",
    );
    // stack
    {
        let mut x = 10;
        x = x + 10;
        println!("x, an int on stack is {} inside the socpe", x);
    }
    // println!("x is {} outisde the socpe", x); //this wont work as x is not more, it was only in the scope

    let x = 32;
    let y = x;
    println!("{} is x and {} is y", x, y); // even after the transfer, x exists unlike the string example below
                                           // this is due to heap moves ownership while in stack creating a copy is so cheap that it makes a copy such that they point to different addresses but with same value

    // HEAP
    // managing memory on heap is difficult
    let string1 = "Something".to_string();
    let string2 = string1;
    // println!("{} was the first string", string1); // this raises error cos data of string1 is now owned by string2
    println!("'{}' was the second string but not in the first", string2); // ownership is move to string2 and string1 does not point to anything

    // to avoid making string1 recaptured, borrowing can be done as a reference or a clone
    let string1 = "Something".to_string();
    let string2 = &string1; // reference
    let string3 = string2.clone(); // clone, a completely new memory is created and the variable points to that, thus avoiding race conditions, thus costly
    println!("'{}' was the first string", string1);
    println!("'{}' was the second string", string2);
    println!("'{}' was the third string", string3);

    let _heap_vector: Vec<i8> = Vec::new(); // can also be done using vec![1,2,3,4,5]
    let _heap_string: String = String::from("Somethin'"); // string can never be in stack, as it is a collection of u8 under the hood and they can grow

    // affect of passing paramters to a function
    let string = String::from("Something");
    function_eats_heap_variables(x, string);
    println!(
        "x with the value in main {} can be reused as it is on stack",
        x
    );
    // println!( "String with the value {} cant be reused as it is on heap", string); // this raises error as "string" is no longer the owner of this memory when it passed it ot the function

    // to make a heap variable return it can be made mutable, then passed into the function such that the function returns the original one back
    let mut string4 = String::from("Something here");
    println!("{} is the variable in main", string4);
    string4 = function_cant_eat_heap(string4);
    println!("{} is the variable in main after function", string4);

    // but this method get complicated if multiple variables are returned
    // BORROWING
    // since only one owner can exist, but multiple references if immutable, almost always, paramters can be passed with reference, this is borrowing where the function takes charge for a while and returns it after use
    // borrowing can be done on stack too
    let string5 = String::from("Something");
    function_borrowing_heap(&string5);
    println!(
        "{} is the variable in main after function with refernce",
        string5
    );

    // string slices
    let string5 = String::from("Something here");
    let _string_slice: &str = "Some"; // a string slice is neither on heap or stack as it is just a pointer

    // one owner multiple references, is possible if the variable promises to be immutable to avoid confusion to references or settle with value before referncing it if it has to be mutable
    // either this or costly clone
    // using this can help in suing parallel processing and utilizing more cores
    let s6 = &string5;
    let s7 = &string5;
    let s8 = &string5;
    println!(
        "'{}' is owned and these are referenced '{}' '{}' '{}'",
        string5, s6, s7, s8
    );

    // string concatenation using reference
    let string6 = "Some";
    let string7 = "thing";
    let str_vec: Vec<&str> = vec![&string6, &string7];
    println!("{}+{} = {}", string6, string7, concatenate_string(str_vec));

    // structs are passed aorund as references as there can be many stack/heap fields in the structs
    // this can be done by cloning the structs too but it has to be put in the debugging marco
    // vairable passsed as reference and can be mutate too
    let mut var_struct = NewStruct {
        int: 32,
        float: 25.,
    };
    print!("{} and {}", var_struct.int, var_struct.float);
    let sum = function_struct(&mut var_struct); // this cant be directly insereted in println ???
    println!(" sum is {} from the function that ignores float value", sum);
}

fn function_struct(arg: &mut NewStruct) -> f32 {
    arg.float = 0.;
    arg.float + arg.int as f32
}

#[derive(Debug, Clone, Copy)] //
struct NewStruct {
    int: i32,
    float: f32,
}

fn concatenate_string(arg: Vec<&str>) -> String {
    [arg[0], arg[1]].concat()
}

fn function_borrowing_heap(variable: &String) {
    println!("{} is the variable in function with reference", variable);
}

fn function_cant_eat_heap(mut heap_variable: String) -> String {
    heap_variable = [heap_variable.to_string(), " here".to_string()].concat();
    println!("{} is the variable in function", heap_variable);
    heap_variable
}

fn function_eats_heap_variables(mut arg_stack: i32, arg_heap: String) -> () {
    println!("argument having {} and in heap is now consumed by the function and can no longer be used in main after passing it here", arg_heap );
    arg_stack = arg_stack + 10; // this makes chages to the variable in the function as it is declared mutable, but the original value in the main is still unchaged, as the memory they are pointing to is different
    println!(
        "argument having {} and in stack of fucntion can be used in main after passing it here",
        arg_stack
    );
}
