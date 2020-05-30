// https://www.youtube.com/watch?v=

/*
> It is asynchronous
> Tokio is used
> blocking has to be enabled in toml

*/

use reqwest;
use reqwest::header;
use reqwest::Error;

fn main() -> Result<(), Error> {
    let res = get_request("https://crates.io/crates/reqwest")?; // propagating it up

    // reading the response into a string
    // let mut body = String::new();
    // res.read_to_string(&mut body).unwrap();
    println!(
        "Status: {:?}\nHeader:{:?}\nBody: {:?}",
        res.status(),
        res.headers(),
        res
    );
    Ok(())
}

fn get_request(url: &str) -> Result<reqwest::blocking::Response, Error> {
    // blocking is the syncronous verison, for async await is used
    let res = reqwest::blocking::get(url)?;
    Ok(res)
}

/*
OUTPUT
Status: 403
Header:{"content-length": "667", "connection": "keep-alive", "server": "nginx", "date": "Sat, 30 May 2020 05:56:44 GMT", "strict-transport-security": "max-age=31536000", "via": "1.1 vegur, 1.1 0c9be32d480a5d5a8aab24b58c540170.cloudfront.net (CloudFront)", "x-cache": "Error from cloudfront", "x-amz-cf-pop": "BLR50-C2", "x-amz-cf-id": "UwQ6RS4BiCxTlZCU43bUNH4kW1yMGjkNKmqKCKO7ID-h9_uAq_iCSQ=="}
Body: Response { url: "https://crates.io/crates/reqwest", status: 403, headers: {"content-length": "667", "connection": "keep-alive", "server": "nginx", "date": "Sat, 30 May 2020 05:56:44 GMT", "strict-transport-security": "max-age=31536000", "via": "1.1 vegur, 1.1 0c9be32d480a5d5a8aab24b58c540170.cloudfront.net (CloudFront)", "x-cache": "Error from cloudfront", "x-amz-cf-pop": "BLR50-C2", "x-amz-cf-id": "UwQ6RS4BiCxTlZCU43bUNH4kW1yMGjkNKmqKCKO7ID-h9_uAq_iCSQ=="} }
 */
