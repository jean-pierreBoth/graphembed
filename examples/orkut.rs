//! example of density decompositon for Orkut graph [https://snap.stanford.edu/data/com-Orkut.html]
//! The graph is undirected with 3072441 nodes and 117 185 083 edges
//! Ground truth communities are provided 

use anyhow::{anyhow};

use std::path::{Path};

use petgraph::prelude::*;

use graphembed::io::csv::unweighted_csv_to_graphmap;

/// Directory containing the 2 data files 
/// TODO use clap in main
const ORKUT_DATA_DIR : &'static str = "/home/jpboth/Data/Graphs/Orkut/";


/// Read graph (given as csv) and ground truth communities
fn read_orkutdir(dirpath : &Path) -> Result<Graph<u32, (), Undirected>, anyhow::Error> {
    let fpath = dirpath.clone().join("com-orkut.ungraph.txt");
    // use csv to unweighted graph_map
    let graphmap = unweighted_csv_to_graphmap::<u32,Undirected>(&fpath, b'\t');
    if graphmap.is_err() {
        std::panic!("cannot open file : {:?}", fpath);
    }
    let graph = graphmap.unwrap().into_graph::<u32>();

    Ok(graph)
} // end of read_orkutdir


pub fn main() {

    let orkutg = read_orkutdir(Path::new(ORKUT_DATA_DIR));

} // end of main