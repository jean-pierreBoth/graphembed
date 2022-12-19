//! example of density decompositon for Orkut graph [https://snap.stanford.edu/data/com-Orkut.html]
//! The graph is undirected with 3072441 nodes and 117 185 083 edges
//! Ground truth communities are provided 


use std::path::{Path};

use petgraph::prelude::*;

use graphembed::prelude::*;

/// Directory containing the 2 data files 
/// TODO use clap in main
const ORKUT_DATA_DIR : &'static str = "/home/jpboth/Data/Graphs/Orkut/";


/// Read graph (given as csv) and ground truth communities
fn read_orkutdir(dirpath : &Path) -> Result<Graph<u32, f64 , Undirected, u32>, anyhow::Error> {
    let fpath = dirpath.clone().join("com-orkut.ungraph.txt");
    // use csv to unweighted graph_map
    log::info!("read_orkutdir : reading {fpath:?}");
    let graphmap = weighted_csv_to_graphmap::<u32 ,f64, Undirected>(&fpath, b'\t');
    log::info!("read_orkutdir : reading {fpath:?}, done");
    if graphmap.is_err() {
        std::panic!("cannot open file : {fpath:?}");
    }
    let graph = graphmap.unwrap().into_graph::<u32>();
    log::info!("graph loaded");
    Ok(graph)
} // end of read_orkutdir


pub fn main() {
    //
    let _ = env_logger::Builder::from_default_env().init();

    let orkut_graph = read_orkutdir(Path::new(ORKUT_DATA_DIR));
    if orkut_graph.is_err() {
        println!("cannot load orkut graph");
    }
    //
    let orkut_graph = orkut_graph.unwrap();
    let nb_iter = 500;
    approximate_decomposition(&orkut_graph , nb_iter);
} // end of main