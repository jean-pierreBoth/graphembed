//! example of density decompositon for Orkut graph [https://snap.stanford.edu/data/com-Orkut.html]
//! 
//! The graph is undirected with 3072441 nodes and 117 185 083 edges
//! 
//! 
//! The purpose of this example is to test coherence between embedded orkut wit communities.
//! We reload embedding , block decomposition dumped by orkut_hnsw example




use anyhow::{anyhow};
use graphembed::embed::tools::jaccard;

use std::path::{Path};
use std::io::{BufReader};
use std::fs::{OpenOptions};
use std::io::prelude::*;
use std::str::FromStr;

use petgraph::prelude::*;

use graphembed::prelude::*;

use graphembed::validation::anndensity::embeddedtohnsw;

use hnsw_rs::prelude::*;
use hnsw_rs::hnswio::*;
use hnsw_rs::prelude::DistPtr;

/// Directory containing the 2 data files 
/// TODO use clap in main
const ORKUT_DATA_DIR : &'static str = "/home/jpboth/Data/Graphs/Orkut/";


/// Read graph (given as csv) and ground truth communities
fn read_orkut_graph(dirpath : &Path) -> Result<Graph<u32, f64 , Undirected, u32>, anyhow::Error> {
    let fpath = dirpath.clone().join("com-orkut.ungraph.txt");
    // use csv to unweighted graph_map
    log::info!("read_orkut_graph : reading {fpath:?}");
    let graphmap = weighted_csv_to_graphmap::<u32 ,f64, Undirected>(&fpath, b'\t');
    log::info!("read_orkutdir : reading {fpath:?}, done");
    if graphmap.is_err() {
        std::panic!("cannot open file : {fpath:?}");
    }
    let graph = graphmap.unwrap().into_graph::<u32>();
    log::info!("graph loaded");
    Ok(graph)
} // end of read_orkutgraph



fn read_orkut_com(dirpath : &Path) -> anyhow::Result<Vec<Vec<usize>>> {
    let fpath = dirpath.clone().join("com-orkut.top5000.cmty.txt");
    log::info!("read_orkut_com : reading {fpath:?}");
    let fileres = OpenOptions::new().read(true).open(&fpath);
    if fileres.is_err() {
        log::error!("read_orkut_com : reload could not open file {:?}", fpath.as_os_str());
        println!("read_orkut_com could not open file {:?}", fpath.as_os_str());
        return Err(anyhow!("read_orkut_com could not open file {}", fpath.display()));            
    }
    let file = fileres?;
    let bufreader = BufReader::new(file);
    let lines = bufreader.lines();
    let mut numline = 0;
    let mut communities = Vec::<Vec<usize>>::with_capacity(5000);
    for line in lines {
        if line.is_err() {
            log::error!("error reading file : {:?} at line : {}",fpath.as_os_str(),numline);
            return Err(anyhow!(" error reading file : {:?} at line : {}",fpath.as_os_str(),numline));
        }
        // split and decode line. line consists in usize separated by a tab
        let line = line.unwrap();
        let splitted : Vec<&str>= line.split('\t').collect();
        let communitiy : Vec<usize> = splitted.iter().map(|s| usize::from_str(*s).unwrap()).collect();
        communities.push(communitiy);
        numline += 1;
    }
    //
    return Ok(communities);
} // end of read_orkut_com



fn jaccard_distance<T:Eq>(v1:&[T], v2 : &[T]) -> f64 {
    assert_eq!(v1.len(), v2.len());
    let common = v1.iter().zip(v2.iter()).fold(0usize, |acc, v| if v.0 == v.1 { acc + 1 } else {acc});
    1.- (common as f64)/(v1.len() as f64)
} // end of jaccard



pub fn main() {
    // TODO clap ...
    let _ = env_logger::Builder::from_default_env().init();

    let orkut_graph = read_orkut_graph(Path::new(ORKUT_DATA_DIR));
    if orkut_graph.is_err() {
        println!("cannot load orkut graph");
    }
    log::info!("orkut graph read");
    //
    let orkut_graph = orkut_graph.unwrap();
    let decomposition : StableDecomposition;
    //
    // check if we have a stored decomposition 
    //
    let dump_path = Path::new("/home/jpboth/Rust/graphembed/orkut-decomposition.json");
    //
    let fileres = OpenOptions::new().read(true).open(&dump_path);
    if fileres.is_err() {
        log::error!("reload could not open file {:?}, will do decomposition", dump_path.as_os_str());
        log::error!("orkut-decomposition.json not found, run orkut_hnsw to be able to reload decomposition");
        panic!("orkut-decomposition.json not found, run orkut_hnsw to be able to reload decomposition");       
    }
    log::info!("found json file for stored decomposition");
    // we reload decomposition
    let res = StableDecomposition::reload_json(dump_path);
    if res.is_err() {
        log::info!("could not reload json decompositon");
        panic!("found orkut decompositon but could not reload it")
    }
    decomposition = res.unwrap();

    let nb_blocks = decomposition.get_nb_blocks();
    log::info!("orkut decomposition got nb_block : {nb_blocks}");
    //
    for blocnum in 0..nb_blocks.min(300) {
        let bsize = decomposition.get_nbpoints_in_block(blocnum).unwrap();
        log::info!("orkhut : points of block : {blocnum} , {bsize}");
    }
    //
    let _communities = read_orkut_com(Path::new(ORKUT_DATA_DIR)).unwrap();
    //
    // now we reload the embedding
    //
    
/*  // if we really need the explicitly the embedding
    let orkut_embedding : Embedding<usize, usize, Embedded<usize>>;
    let orkut_bson_path= Path::new("/home/jpboth/Rust/graphembed/orkut_embedded.bson");
    let fileres = OpenOptions::new().read(true).open(&orkut_bson_path);
    if fileres.is_err() {
        log::info!("cannot reoad orkut embedding, did not find file : {:?}", orkut_bson_path);
        std::panic!("cannot reoad orkut embedding, did not find file : {:?}", orkut_bson_path);
    }
    else {
        log::info!("found bson file, reloading embedding, trying to reload from {:?}", &orkut_bson_path);
        let reloaded = embeddedbson::bson_load::<usize, usize, Embedded<usize>>(orkut_bson_path.to_str().unwrap());
        if reloaded.is_err() {
            log::error!("reloading of bson from {:?} failed", &orkut_bson_path);
            log::error!("error is : {:?}", reloaded.err());
            std::panic!("bson reloading failed");
        }
        let bson_reloaded = reloaded.unwrap();
        let embedding = from_bson_with_jaccard(bson_reloaded);
        orkut_embedding = embedding.unwrap();
    } */

    //
    // we reload the hnsw dumped by example orkut_hnsw
    //
    let graphfname = String::from("dumpreloadtestgraph.hnsw.graph");
    let graphpath = Path::new(&graphfname);
    let graphfileres = OpenOptions::new().read(true).open(graphpath);
    if graphfileres.is_err() {
        println!("test_dump_reload: could not open file {:?}", graphpath.as_os_str());
        std::panic::panic_any("test_dump_reload: could not open file".to_string());            
    }
    let graphfile = graphfileres.unwrap();
    //  
    let datafname = String::from("dumpreloadtestgraph.hnsw.data");
    let datapath = Path::new(&datafname);
    let datafileres = OpenOptions::new().read(true).open(&datapath);
    if datafileres.is_err() {
        println!("test_dump_reload : could not open file {:?}", datapath.as_os_str());
        std::panic::panic_any("test_dump_reload : could not open file".to_string());            
    }
    let datafile = datafileres.unwrap();
    //
    let mut graph_in = BufReader::new(graphfile);
    let mut data_in = BufReader::new(datafile);
    // we need to call load_description first to get distance name
    let hnsw_description = load_description(&mut graph_in).unwrap();
    let mydist = DistPtr::<usize,f64>::new(jaccard_distance::<usize>);

    let hnsw_loaded : Hnsw<usize, DistPtr<usize,f64> >= load_hnsw_with_dist(&mut graph_in, &hnsw_description,  mydist, &mut data_in).unwrap();

    // now we can check how are embedded blocks and communities we examined in Notebook
} // end of main