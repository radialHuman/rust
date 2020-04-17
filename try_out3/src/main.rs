mod cargo;
#[allow(dead_code)] // to allow calling in other files without using them
mod casting_shadowing_constant_static;
mod conditionals;
mod enums;
mod functions_procedures;
mod lib_main;
mod ownership_borrowing;
mod primitives; // to call in other files
mod println_format;
mod strings;
mod structs_traits;
mod tuple;

fn main() {
    primitives::code(); // to call functions inside the file called above
    strings::code();
    functions_procedures::code();
    conditionals::code();
    tuple::code();
    structs_traits::code();
    enums::code();
    ownership_borrowing::code();
    casting_shadowing_constant_static::code();
    println_format::code();
    lib_main::code();
    cargo::code();
}
