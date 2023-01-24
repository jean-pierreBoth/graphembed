//! example of density decompositon for Orkut graph [https://snap.stanford.edu/data/com-Orkut.html]
//! 
//! The graph is undirected with 3072441 nodes and 117 185 083 edges
//! 
//! 
//! The purpose of this example is to test coherence between embedded orkut wit communities.
//! We reload embedding , block decomposition dumped by orkut_hnsw example




use anyhow::{anyhow};

use std::path::{Path};
use std::fs::{OpenOptions};
use std::io::prelude::*;
use std::str::FromStr;

use std::io::{BufReader, BufWriter };

use petgraph::prelude::*;
use petgraph::stable_graph::DefaultIx;

use graphembed::prelude::*;
use graphembed::validation::anndensity::*;

use rand::{Rng};

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


// receive base name of dump. Files to reload are made by concatenating "hnsw.data" and "hnsw.graph" to path
fn reload_orkut_hnsw(path : String) -> Hnsw<usize, DistPtr<usize,f64> > {
    // 
    let mut graphfname = path.clone();
    graphfname.push_str(".hnsw.graph");
    let graphpath = Path::new(&graphfname);
    let graphfileres = OpenOptions::new().read(true).open(graphpath);
    if graphfileres.is_err() {
        println!("test_dump_reload: could not open file {:?}", graphpath.as_os_str());
        std::panic::panic_any("test_dump_reload: could not open file".to_string());            
    }
    let graphfile = graphfileres.unwrap();
    //  
    let mut datafname = path.clone();
    datafname.push_str(".hnsw.data");
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
    //
    hnsw_loaded
}  // end of reload_orkut_hnsw


fn block_dump_json(block_path : &Path, block_array : &ndarray::Array2<f32>) {
    let fileres = OpenOptions::new().write(true).create(true).truncate(true).open(&block_path);
    if fileres.is_err() {
        log::error!("block_matrix dump : dump could not open file {:?}", block_path.as_os_str());
        println!("block matix dump: could not open file {:?}", block_path.as_os_str());
    }
    // 
    let mut writer = BufWriter::new(fileres.unwrap());
    let _ = serde_json::to_writer(&mut writer, &block_array).unwrap();
}  // end of block_dump_json




fn block_load_json(block_path : &Path) -> ndarray::Array2<f32> {
    let fileres = OpenOptions::new().read(true).create(true).truncate(true).open(&block_path);
    if fileres.is_err() {
        log::error!("block_matrix dump : dump could not open file {:?}", block_path.as_os_str());
        println!("block matix dump: could not open file {:?}", block_path.as_os_str());
    }
    let loadfile = fileres.unwrap();
    let reader = BufReader::new(loadfile);
    let block_transition : ndarray::Array2<f32> = serde_json::from_reader(reader).unwrap();
    block_transition
} // end of block_load_json



// try to examine what happens to community through the embedding
// check mean distance between couples inside the community? versus outside?
// possibly check also mean distance between neighbours
fn analyze_community(community : &Vec<usize>, orkut_embedding : &Embedding<usize, usize, Embedded<usize>>, nb_sample : usize) -> f64 {
    // loop , sample couple, compute distance
    let mut dist_in_com = Vec::<f64>::with_capacity(nb_sample);
    //
    let mut rng = rand::thread_rng();
    let com_size = community.len();
    
    for _ in 0..nb_sample {
        let idx1 = rng.gen_range(0..com_size);
        let idx2 = loop {
            let idx2 = rng.gen_range(0..com_size);
            if idx1 != idx2 {
                break idx2;
            }
        };
        // compute dist between the 2 nodes
        let dist = orkut_embedding.get_node_distance(community[idx1], community[idx2]);
        dist_in_com.push(dist);
    }
    //
    let mean_dist = dist_in_com.iter().sum::<f64>()/ dist_in_com.len() as f64;
    //
    log::info!("mean distance between couples : {:.3e}", mean_dist);
    //
    mean_dist
} // end of analyze_community



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
    let decomposition = res.unwrap();

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
    
    // if we really need the explicitly the embedding
    let orkut_embedding : Embedding<usize, usize, Embedded<usize>>;
    let orkut_bson_path= Path::new("/home/jpboth/Rust/graphembed/orkut_embedded.bson");
    let fileres = OpenOptions::new().read(true).open(&orkut_bson_path);
    if fileres.is_err() {
        log::info!("cannot reoad orkut embedding, did not find file : {:?}", orkut_bson_path);
        std::panic!("cannot reload orkut embedding, did not find file : {:?}", orkut_bson_path);
    }
    else {
        log::info!("found bson file, reloading embedding, trying to reload from {:?}", &orkut_bson_path);
        let reloaded = bson_load::<usize, usize, Embedded<usize>>(orkut_bson_path.to_str().unwrap());
        if reloaded.is_err() {
            log::error!("reloading of bson from {:?} failed", &orkut_bson_path);
            log::error!("error is : {:?}", reloaded.err());
            std::panic!("bson reloading failed");
        }
        let bson_reloaded = reloaded.unwrap();
        let embedding = from_bson_with_jaccard(bson_reloaded);
        orkut_embedding = embedding.unwrap();
    }

    //
    // we reload the hnsw dumped by example orkut_hnsw
    //
    let hnsw_loaded : Hnsw<usize, DistPtr<usize,f64> >= reload_orkut_hnsw(String::from("orkuthnsw"));
    //
    let d_res = density_analysis::<usize, DistPtr<usize,f64>, DefaultIx>(&orkut_graph,
                                    orkut_embedding.get_embedded_data(), 
                                    Some(hnsw_loaded), Some(decomposition));
    if d_res.is_err() {
        log::error!("density analysis failed with error: {:?}", d_res.as_ref().err());
        std::process::exit(1);
    }
 
    log::info!("\n\n dumping block array");
    let block_array = d_res.unwrap();
    log::info!(" block_array : {:?}", &block_array);
    let block_path= Path::new("orkut_block_mat.json");
    block_dump_json(block_path, &block_array);
    //
    log::info!("exiting from orkut_check");
    //
    // now we can check how are embedded blocks and communities we examined in Notebook
    //

} // end of main