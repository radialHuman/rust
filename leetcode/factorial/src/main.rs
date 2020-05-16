fn main() {

	// let number = 50;
    // fn_factorial(number);
    let number = 50;
    let factorial = recurssive_factorial(number);
    println!("Result for {} from recurssive_factorial {:?}", number,factorial );

}

// fucntion to find factorial
// fn fn_factorial(n:i32) {
// 	let mut result = 1;
// 	if n>0 {
// 		for i in 1..n
// 			{result *= i;}
// 		println!("The factorial of {} is {:?}", n ,result);
// 	}
// 	if n==0{
// 		println!("The factorial of {} is {:?}", n ,result);
// 	}
// 	else {
// 		println!("{:?} doesnt have a factorial", n);
// 	}
// }

fn recurssive_factorial(n:i32)-> i32{
	if n<2{
		1
	}
	else {
		recurssive_factorial(n-1)*n
	}
}
