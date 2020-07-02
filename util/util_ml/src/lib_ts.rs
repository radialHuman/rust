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

4. best_fit_line : returns intercept and slope of the best fit line
    > x: &Vec<f64>
    > y: &Vec<f64>)
     = (f64, f64) 
    
5. pacf : returns a vector of values as in a graph
    > ts: &Vec<f64>
    > lag: usize)
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
    let mae = mean(&ts[1..].to_vec().iter().zip(output[..ts.len()-1].to_vec().iter()).map(|(a,b)| (a-b).abs()).collect());
    println!("Mean square error of this forecasting : {:?}", mse);
    println!("Mean absolute error of this forecasting : {:?}", mae);
    exp_ma
}


pub fn best_fit_line(x: &Vec<f64>, y: &Vec<f64>) -> (f64, f64) {
    // https://pythonprogramming.net/how-to-program-best-fit-line-machine-learning-tutorial/
    // intercept , slope
    let xy = x
        .iter()
        .zip(y.iter())
        .map(|a| a.0 * a.1)
        .collect::<Vec<f64>>();
    let xx = x
        .iter()
        .zip(x.iter())
        .map(|a| a.0 * a.1)
        .collect::<Vec<f64>>();
    let m = ((mean(x) * mean(y)) - mean(&xy)) / ((mean(x) * mean(x)) - mean(&xx));

    let b = mean(y) - m * mean(x);
    (b, m)
}

pub fn pacf(ts: &Vec<f64>, lag: usize) -> Vec<f64> {
    /*
    Unlike ACF, which uses combined effect on a value, here the impact is very specific
    The coeeficient is specific to the lagged value
    This is more useful than ACF as it remoevs the influence of values in between
    */
    // data: https://www.ncdc.noaa.gov/teleconnections/enso/indicators/soi/data.csv
    // https://www.youtube.com/watch?v=ZjaBn93YPWo, http://rinterested.github.io/statistics/acf_pacf.html, https://towardsdatascience.com/understanding-partial-auto-correlation-fa39271146ac
    /*
    STEPS:
    1. While shifting the ts from front, and residual from last (residual is same as ts initially)
    2. From the best fit line between ts and residual find the new residual
    3. From now on best fit line will be found on the residual
    4. The correlation between shifted ts and point residual at each shift will be captured as pacf at that point
    */
    let mut pacf = vec![1.]; // as correlation with it self is 1
    let mut residual = ts.clone();
    for i in 1..lag {
        let mut res_shift = &residual[i..].to_vec();
        // finding correlation
        let corr = correlation(&ts[..(ts.len() - i)].to_vec(), &res_shift, "p");
        pacf.push(corr);
        // calculting best fit line
        let (intercept, slope) =
            best_fit_line(&ts[..(ts.len() - i)].to_vec(), &residual[i..].to_vec());
        // calculating estimate
        let estimate = &ts[..(ts.len() - i)]
            .to_vec()
            .iter()
            .map(|a| (a * slope) + intercept)
            .collect::<Vec<f64>>();
        // modifying residual to act as source data in the next lag
        res_shift = &res_shift
            .iter()
            .zip(estimate.iter())
            .map(|a| a.0 - a.1)
            .collect::<Vec<f64>>();
        println!("slope : {:?}, intercept : {:?}", slope, intercept);
    }

    pacf
}
