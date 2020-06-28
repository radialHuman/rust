/*
DESCRIPTION
-----------------------------------------
STRUCTS
-------

FUNCTIONS
---------
1. autocorrelation : at a given lag
    > 1. ts : &Vec<f64>
    > 2. lag : usize 
    = Result<f64, std::io::Error>

2. simple_ma
    > 1. ts : &Vec<f64>
    > 2. lag : usize 
    = Vec<f64>

3. exp_ma
    > 1. ts : &Vec<f64>
    > 2. alpha : f64 
    = Vec<f64>
*/

pub fn acf(ts: &Vec<f64>, lag: usize) -> Result<f64, std::io::Error> {
    /*
    To check for randomness of time series
    To check if the values are dependent on its past value
    */
    // https://www.itl.nist.gov/div898/handbook/eda/section3/eda35c.htm
    let mean = mean(ts);
    let mut numerator = 0.;
    let mut denominator = 0.;
    for i in 0..ts.len() - lag {
        if i > lag {
            numerator += (ts[i] - mean) * (ts[i - lag] - mean);
            denominator += (ts[i] - mean) * (ts[i] - mean);
        }
    }
    match denominator {
        x if x != 0. => { if ((numerator / denominator).abs() > 0.5) && (lag!=0){
            print!("At {:?} lag the series seems to be correlated\t",lag)
        }
            Ok(numerator / denominator)},
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Denominator is 0!",
        )),
    }
}


pub fn simple_ma(ts: &Vec<f64>, lag: usize) -> Vec<f64>{
    let mut output = vec![];
    for i in 0..lag{
        if lag+i<= ts.len()
        {let sub_ts = ts[i..lag+i].to_vec();
        output.push(sub_ts.iter().fold(0.,|a,b| a+b)/sub_ts.len() as f64);}
    }
    pad_with_zero(&mut output, lag, "pre")
}


pub fn exp_ma(ts: &Vec<f64>,  alpha:f64) -> Vec<f64>{
    // https://www.youtube.com/watch?v=k_HN0wOKDd0
    // assuming first forecast is == first actual value
    // new forecast  = aplha*actual old value+(1-alpha)*forecasted old value
    let mut output = vec![ts[0]];
    for (n,i) in ts[1..].to_vec().iter().enumerate(){
        output.push(alpha*i+(1.-alpha)*output[n]);
    }
    // removing the last value and adding 0 in front
    let exp_ma = pad_with_zero(&mut output[..ts.len()-1].to_vec(), 1, "pre");
    let mse = mean(&ts[1..].to_vec().iter().zip(output[..ts.len()-1].to_vec().iter()).map(|(a,b)| (a-b)*(a-b)).collect());
    println!("Mean square error of this forecasting : {:?}", mse);
    exp_ma
}