//! lib target

use env_logger::Builder;

#[macro_use]
extern crate lazy_static;

// used in mod tests to access some data sets. To be modified according to installation
#[allow(unused)]
static DATADIR: &str = "/home/jpboth/Rust/graphembed/Data";

lazy_static! {
    static ref LOG: u64 = init_log();
}

// install a logger facility
fn init_log() -> u64 {
    Builder::from_default_env().init();
    println!("\n ************** initializing logger *****************\n");
    1
}

pub mod io;

pub mod embed;

pub mod embedding;

pub mod validation;

pub mod prelude;
//

pub mod structure;
#[cfg(feature = "python")]
pub mod python;