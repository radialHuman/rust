fn main_() {
    let (header, values) =
        read_csv("..//util//util_ml//data\\data_banknote_authentication.txt".to_string());

    println!("{:?}", train_test_split(&values, 0.25));
}
