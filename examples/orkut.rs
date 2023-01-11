//! example of density decompositon for Orkut graph [https://snap.stanford.edu/data/com-Orkut.html]
//! The graph is undirected with 3072441 nodes and 117 185 083 edges
//! The output is the ordered list of densest stable sets
//! 
//! Frank-Wolfe 500 iterations run in 12.3mn. The whole computation need 13.4mn on a 8 cores 2-thread/core laptop with i7 intel proc

use anyhow::{anyhow};

use std::path::{Path};
use std::io::{BufReader};
use std::fs::{OpenOptions};
use std::io::prelude::*;
use std::str::FromStr;

use petgraph::prelude::*;

use graphembed::prelude::*;
use graphembed::io::*;


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





pub fn main() {
    // TODO clap ...
    let _ = env_logger::Builder::from_default_env().init();

    let orkut_graph = read_orkut_graph(Path::new(ORKUT_DATA_DIR));
    if orkut_graph.is_err() {
        println!("cannot load orkut graph");
    }
    //
    let orkut_graph = orkut_graph.unwrap();
    let decomposition : StableDecomposition;
    let nb_iter = 500;
    //
    // check if we have a stored decomposition 
    //
    let dump_path = Path::new("/home/jpboth/Rust/graphembed/orkut-decomposition.json");
    //
    let fileres = OpenOptions::new().read(true).open(&dump_path);
    if fileres.is_err() {
        log::error!("reload could not open file {:?}, will do decomposition", dump_path.as_os_str());
        println!(" reload could not open file {:?}, will do decomposition ", dump_path.as_os_str());
        decomposition = approximate_decomposition(&orkut_graph , nb_iter);
        // and dump decomposition
        let res = decomposition.dump_json(&dump_path);
        match res {
            Ok(_) => { log::info!("orkut decomposition dumped in {dump_path:?} : Ok") ; },
            Err(_) => { log::info!("orkut decomposition dump failed in {dump_path:?} : Err") ; },
        };
        let nb_blocks = decomposition.get_nb_blocks();
        log::info!("orkut decomposition got nb_block : {nb_blocks}");        
    }
    else {
        log::info!("found json file for stored decomposition");
        // we reload decomposition
        let res = StableDecomposition::reload_json(dump_path);
        if res.is_err() {
            log::info!("could not reload json decompositon");
            panic!("found orkut decompositon but could not reload it")
        }
        decomposition = res.unwrap();
    }
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
    // now we embed the graph
    //
    let orkut_embedded : Embedding<usize, usize, Embedded<usize>>;
    let orkut_bson_path= Path::new("/home/jpboth/Rust/graphembed/orkut_emededded.bson");
    let fileres = OpenOptions::new().read(true).open(&orkut_bson_path);
    if fileres.is_err() {
        log::info!("going to embed orkut graph");
        let path = Path::new(ORKUT_DATA_DIR).clone();
        // TODO avoid reloading. need a conversion trimat to graphmap
        let res = csv_to_trimat_delimiters::<f64>(&path.clone().join("com-orkut.ungraph.txt"), false);
        let (orkut_trimat, node_index) = res.unwrap();
        let sketch_size = 200;
        let decay = 0.2;
        let nb_iter = 5;
        let parallel = true;
        let symetric = true;
        let params = NodeSketchParams{sketch_size, decay, nb_iter, symetric, parallel};
        let mut nodesketch = NodeSketch::new(params, orkut_trimat);
        let sketch_embedding = Embedding::new(node_index, &mut nodesketch);
        if sketch_embedding.is_err() {
            log::error!("embedding orkut failed in Embedding::new");
            assert_eq!(1, 0);        
        }
        orkut_embedded = sketch_embedding.unwrap();
        // now we can do a bson dump
        log::info!("dumping orkut embedding ...");
        let output =  output::Output::new(output::Format::BSON, true, &Some(String::from("orkut_emededded")) );
        let bson_res = embeddedbson::bson_dump(&orkut_embedded, &output);
        if bson_res.is_err() {
            log::error!("bson dump in file {} failed", &output.get_output_name());
            log::error!("error returned : {:?}", bson_res.err().unwrap());
            assert_eq!(1,0);
        } 
        log::info!("orkut embedding done");
    }
    else {
        log::info!("found bson file, reloading embedding, trying to reload from {:?}", &orkut_bson_path);
        let reloaded = embeddedbson::bson_load::<usize, usize, Embedded<usize>>(orkut_bson_path.to_str().unwrap());
        if reloaded.is_err() {
            log::error!("reloading of bson from {:?} failed", &orkut_bson_path);
        }
        let _reloaded = reloaded.unwrap();
        // TODO conversion from EmbeddedBsonReload to Embedding
    }

    //
    // we compute hnsw on embedding
    //
} // end of main