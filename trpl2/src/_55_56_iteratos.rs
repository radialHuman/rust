//

/*
Iterators
*/

pub fn function() {
    /*
    > To traverse through a list like collection
    > Iteretor is lazy
    > the trait of iterator has a type which is returned as Option using next
    > iter : immutable reference
    > into_iter() : takes ownership
    > iter_mut() : mutable reference
    */
    let v1 = vec![1, 5, 2, 5, 6, 7];
    let v1_iter = v1.iter();
    // does nothign so far, only when something is called for it acts (lazy)
    for i in v1_iter {
        println!("{:?}", i);
    }
    println!("Next value is {:?}", v1.iter().next());

    // Consuming iterators
    /*
    > Few commands chnage the iterator by removing/aggregating so on like:
    > next()
    > sum()
    */
    let v1 = vec![1., 5., 2., 5., 6., 7.];
    let add: f64 = v1.iter().sum();
    println!("Sum of {:?} is {}", v1, add);

    // Adpator iterators
    /*
    > These functions change the collection into another
    > but nothing happens untill its collected (lazy)
    */
    let new_v1: Vec<_> = v1.iter().map(|x| x / 10.).collect();
    println!("A 10th of {:?} is {:?}", v1, new_v1);

    // using envronment variable sin clousers of iterators
    let threshold = 5.;
    let smaller_v1: Vec<_> = v1.iter().filter(|x| x > &&threshold).collect(); // threshold gets called even though outside the scope
    println!(
        "Numbers higher than threshold in {:?} are {:?}",
        v1, smaller_v1
    );

    // Customizing iterator trait is possible
}

/*
OUTPUT
1
5
2
5
6
7
Next value is Some(1)
Sum of [1.0, 5.0, 2.0, 5.0, 6.0, 7.0] is 26
A 10th of [1.0, 5.0, 2.0, 5.0, 6.0, 7.0] is [0.1, 0.5, 0.2, 0.5, 0.6, 0.7]
Number higher than threshold in [1.0, 5.0, 2.0, 5.0, 6.0, 7.0] are [6.0, 7.0]
*/
