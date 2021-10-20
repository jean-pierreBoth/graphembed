//! Construct or dump a (small) graph from data from a csv


use std::fmt::{Debug};

use log::*;
use anyhow::{anyhow, Error};


use std::fs::OpenOptions;
use std::path::{Path};

use std::io::{Write,BufWriter, BufReader};

use csv::ReaderBuilder;

use petgraph::graph::{Graph, IndexType};
use petgraph::{Directed, Undirected, EdgeType};



// From csv file we have nodes indexed by u32, edge multiplicity as f32,  edge can be negative 
// (see bitcoin files  https://snap.stanford.edu/data/index.html)
//
// https://snap.stanford.edu/data/index.html
// p2p-Gnutella09.txt  peer to peer directed unweighted
// ca-GrQc.txt  collaboration network undirected

/// Ty is Directed (default) or UnDirected
/// Ix is the node and edge index type , default u32
/// N and E in Grap<N,E,Ty,Ix are data associated to node and edge respectively>
/// instantiate with UnDirected for undirected graph
/// 
pub fn directed_from_csv<N, E,Ix>(filepath : &Path) -> anyhow::Result<usize> 
    where Ix : IndexType {
    //
    // initialize a reader from filename, skip lines beginning with # or %
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("ProcessingState reload_json : reload could not open file {:?}", filepath.as_os_str());
        println!("ProcessingState reload_json: could not open file {:?}", filepath.as_os_str());
        return Err(anyhow!("ProcessingState reload_json could not open file"));            
    }
    let file = fileres.unwrap();
    let mut rdr = ReaderBuilder::new().flexible(false).from_reader(file);
    //
    let graph = Graph::<N,E, Directed, Ix>::with_capacity(10_000, 100_000);
    //
    let mut nb_record = 0;
    for result in rdr.records() {
        let record = result?;
        nb_record += 1;
        if log::log_enabled!(Level::Info) {
            log::debug!("{:?}", record);
        }

    }
    Ok(1)
} // end of from_csv


#[cfg(test)]

mod tests {

fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn load_undirected() {
    // 
    println!("\n\n load_undirected");
    log_init_test();

} // end test load_undirected

#[test]
fn load_directed_unweighted() {
    // We load CA-GrQc.txt taken from Snap data directory
    println!("\n\n load_directed");
    log_init_test();

} // end test load_undirected


}  // edn of mod tests
