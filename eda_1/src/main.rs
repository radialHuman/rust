use csv::Reader;
fn main() {
    // reading in the data
    let mut reader = csv::Reader::from_path("Automobile_data.csv");
    for i in reader {
        println!("{:?}", i);
    }
}
