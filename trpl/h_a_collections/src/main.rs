use std::collections::HashMap;

fn main() {
    // heap storage, hence dynamic in nature
    // vector, string, hashmap
    println!("\n**** VECTOR ****");
    // creating using generics
    let mut v: Vec<i32> = Vec::new(); // needs type declaration
    println!("Length of a new vector is {}", v.len());
    v.push(32);
    v.push(2);
    v.push(3);
    for i in v.iter() {
        println!("{}", i);
    }

    // creating directly
    let v2 = vec![1, 5, 7, 2, 8, 2, 5, 3]; // does not require type as it can infer
    println!("The new vector is ");
    for i in v2.iter() {
        print!("{}, ", i);
    }
    println!();

    // concatenating two vectors
    let mut v1 = [&v[..], &v2[..5]].concat();
    println!("The concatenated vector is ");
    for i in v1.iter() {
        print!("{}, ", i);
    }
    println!();
    println!("{} is the 3rd element in the concatenated vector", &v1[4]);
    println!(
        "{:?} is the 4th element in the concatenated vector",
        v1.get(5)
    ); // ??? why is this some and

    // using reference
    for i in &v1 {
        print!("{}, ", i);
    }
    println!();
    // using mutable reference
    for i in &mut v1 {
        print!("{}, ", *i * 2);
    }
    println!("",);
    // using enums in vector
    println!("\n**** ENUMS IN VECTORS ****",);
    // even though vectors can have only same type od data in it, enums can be used to bend that rule
    let e_vec = vec![
        E1::Name("Some One".to_string()),
        E1::Age(23),
        E1::Male(false),
    ];
    println!(
        "{:?} is {:?} and is a male : {:?}",
        e_vec[0], e_vec[1], e_vec[2]
    );
    println!("\n**** STRINGS ****",);
    // default is a string literal which is a string slice str
    // std lib allows String and others like OsString, CString, OsStr, CStr
    // made by UTF-8
    // complicated data structure
    // 2 ways to create
    let s1 = "Something".to_string();

    let s2 = String::from(" exists");
    // two ways to concatenate
    let mut s3 = s1 + &s2;
    // println!("{}", s1); // this cant happen as s1 is moved to s3
    println!("{}", s3);

    s3.push_str(" there");
    println!("{}", s3);

    s3.push('!');
    println!("{}", s3);

    // string cant be indexed
    // println!("{}", s3[3]); // this wont happen due to utf-8 way of storing bytes, and its performance
    println!("{}", &s3[..10]);

    // iterating over string
    for i in s3.chars() {
        print!("{}|", i);
    }
    println!();

    for i in s3.bytes() {
        print!("{}|", i);
    }
    println!();

    println!("\n**** HASHMAPS ****",);
    // std library has it ehence calling it in the beginning
    // least use so less support
    // no macro to create them easily like vec or string
    // ALl key and values must of same type
    // values which are from heap gets moved like string, while from stcka gets copied like i32
    let mut dict1 = HashMap::new();
    dict1.insert(String::from("Name1"), "Some One".to_string());
    dict1.insert(String::from("Name2"), "Some Two".to_string());

    // another way is
    let keys = vec![String::from("Age1"), String::from("Age2")];
    let values = vec![23, 34];

    let mut dict2: HashMap<_, _> = keys.iter().zip(values.iter()).collect();
    println!("{:?} is in HashMap", dict2.get(&String::from("Age2")));

    // iterating over it
    println!("The hashmap was:");
    for (k, v) in &dict1 {
        println!("{}:{}", k, v);
    }

    dict1.insert(String::from("Name1"), "Some Zero".to_string());
    println!("The hashmap is:");
    for (k, v) in &dict1 {
        println!("{}:{}", k, v);
    }

    // check if key exists else put something else
    dict1
        .entry(String::from("Name1"))
        .or_insert("Some One".to_string());
}

#[derive(Debug)]
enum E1 {
    Name(String),
    Age(i32),
    Male(bool),
}

/*
OUTPUT
**** VECTOR ****
Length of a new vector is 0
32
2
3
The new vector is
1, 5, 7, 2, 8, 2, 5, 3,
The concatenated vector is
32, 2, 3, 1, 5, 7, 2, 8,
5 is the 3rd element in the concatenated vector
Some(7) is the 4th element in the concatenated vector
32, 2, 3, 1, 5, 7, 2, 8,
64, 4, 6, 2, 10, 14, 4, 16,

**** ENUMS IN VECTORS ****
Name("Some One") is Age(23) and is a male : Male(false)

**** STRINGS ****
Something exists
Something exists there
Something exists there!
Something
S|o|m|e|t|h|i|n|g| |e|x|i|s|t|s| |t|h|e|r|e|!|
83|111|109|101|116|104|105|110|103|32|101|120|105|115|116|115|32|116|104|101|114|101|33|

**** HASHMAPS ****
Some(34) is in HashMap
The hashmap was:
Name2:Some Two
Name1:Some One
The hashmap is:
Name2:Some Two
Name1:Some Zero
*/
