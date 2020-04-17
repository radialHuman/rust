pub fn code() {
    print!("**** MAIN ****",);
    println!(
        "
> it is runnable by itself
> it creates an executable binary
> must be main.rs in src
> can import crates"
    );
    print!("**** LIB ****",);
    println!(
        "
> must be lib.rs in src
> can be same as main but with a different name
> Not runnable on its own
> can be complied using build but not run using run
> like a module it can contain fucntions and structs to be called by other files
> Nice for abstraction and modularity"
    );

    // package is a dev. project, can be main or lib
    // Crate can be complied and used
    //
}

/*
OUTPUT
**** MAIN ****
> it is runnable by itself
> it creates an executable binary
> must be main.rs in src
> can import crates
*/
