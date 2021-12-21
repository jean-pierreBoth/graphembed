//! lib target


use env_logger::{Builder};

#[macro_use]
extern crate  lazy_static;

lazy_static! {
    static ref LOG: u64 = {
        let res = init_log();
        res
    };
}

// install a logger facility
fn init_log() -> u64 {
    Builder::from_default_env().init();
    println!("\n ************** initializing logger *****************\n");    
    return 1;
}

pub mod io;

pub mod atp;

pub mod sketching;

// pub mod sketching;