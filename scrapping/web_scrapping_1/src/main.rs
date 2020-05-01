/*
 * To extract links for the books available
 * And download them
 */

// #[tokio::main]
// async fn main() -> Result<(), Box<dyn std::error::Error>> {
//     let resp = reqwest::get("https://httpbin.org/ip").await?.text().await?;
//     println!("{:#?}", resp);
//     Ok(())
// }

// Source: https://github.com/Coding-Runner/Project

use regex::Regex;
use reqwest;
use scraper::{Html, Selector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let req = reqwest::get("https://towardsdatascience.com/springer-has-released-65-machine-learning-and-data-books-for-free-961f8181f189")?.text()?;
    // let body = Html::parse_document(&req);
    // println!("{:?}", req);
    let re =
        Regex::new(r"[0-9][0-9][0-9]-[0-9]*-[0-9][0-9][0-9]*-[0-9][0-9][0-9][0-9]*-[0-9]...a")?;

    // making a list of links to visit downlaod page
    let mut list_of_links = vec![];
    for i in re.find_iter(req.as_str()) {
        let y: Vec<&str> = i.as_str().split("\"").collect();
        let x = y[0];
        list_of_links.push(format!(
            "{}{}{}",
            // "https://link.springer.com/openurl?genre=",
            "https://link.springer.com/content/pdf/10.1007%2F",
            x,
            ".pdf"
        ));
    }
    println!("{:?}, {}", list_of_links, list_of_links.len());
    Ok(())
}
