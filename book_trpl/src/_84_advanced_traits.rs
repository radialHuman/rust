// 19.2

/*
*/

use std::ops::Add;

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

impl Add for Point {
    type Output = Point; // associative type, placeholder
    fn add(self, other: Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

pub fn function() {
    // Why Trait is better than generics ???

    // Operator overloading
    let point1 = Point { x: 3, y: 5 };
    let point2 = Point { x: 2, y: 4 };
    let sum = &point1.add(point2);
    println!("{:?}", sum);
}

/*
OUTPUT
*/
