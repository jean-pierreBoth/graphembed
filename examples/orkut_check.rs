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

use rand::distributions::{Distribution, Uniform};

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



fn read_orkut_com(dirpath : &Path) -> anyhow::Result<Vec<Vec<u32>>> {
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
    let mut communities = Vec::<Vec<u32>>::with_capacity(5000);
    for line in lines {
        if line.is_err() {
            log::error!("error reading file : {:?} at line : {}",fpath.as_os_str(),numline);
            return Err(anyhow!(" error reading file : {:?} at line : {}",fpath.as_os_str(),numline));
        }
        // split and decode line. line consists in usize separated by a tab
        let line = line.unwrap();
        let splitted : Vec<&str>= line.split('\t').collect();
        let mut community : Vec<u32> = splitted.iter().map(|s| u32::from_str(*s).unwrap()).collect();
        community.sort_unstable();
        assert!(community[0] != 0);
        communities.push(community);
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
// compares edges inside community with edges going out of community
// possibly : which edges in are matched in hnsw?
fn analyze_community(community : &Vec<u32>, graph : &Graph<u32, f64, Undirected>, 
            orkut_embedding : &Embedding<usize, usize, Embedded<usize>>, nb_sample : usize) -> (f64, f64) {
    // is_sorted is unstable
    // assert!(community.is_sorted());
    // loop , sample couple, compute distance
    let mut dist_in_com = Vec::<f64>::with_capacity(nb_sample);
    let mut dist_out_com = Vec::<f64>::with_capacity(nb_sample);
    //
    let _node_indexation = get_graph_indexation(graph);
    //
    let uniform = Uniform::<f64>::new(0., 1.);
    let mut rng = rand::thread_rng();
    // loop with acceptance rejection on nodes.
    let com_size = community.len();
    let ratio = nb_sample as f64 / com_size as f64;
    log::info!("acceptance ratio is : {:.3e}", ratio);

    for i in 0..com_size {
        // recall that community contains nodes names (type N in Graph<N,   >) and note indices!
        let node = community[i];
        assert!(node != 0);
        // iterate over neighbours of each node
        let mut neighbours = graph.neighbors_undirected(NodeIndex::new(node as usize));
        while let Some(nidx) = neighbours.next() {
            let xsi = uniform.sample(&mut rng);
            if xsi < ratio {
                // now  nidx is a NodeInddex is neighbour n in community we must get its name
                let n_name = graph[nidx];
                if n_name == 0 {
                    log::info!("n : {:?}", nidx);
                    assert!(n_name != 0);
                }
                let is_in = community.contains(&n_name);
                log::trace!("dist node1 : {:?} node2 : {:?}", node, n_name);
                let dist = orkut_embedding.get_node_distance(node as usize, n_name as usize);
                if is_in {
                    dist_in_com.push(dist);
                }
                else {
                    dist_out_com.push(dist);
                }  
            }   
        }
    }
    //
    let mean_in_dist = dist_in_com.iter().sum::<f64>()/ dist_in_com.len() as f64;
    let mean_out_dist = dist_out_com.iter().sum::<f64>()/ dist_out_com.len() as f64;
    //
    log::info!("mean distance between neighbours, in : {:.3e} len : {:?}, out : {:.3e} len : {:?}", mean_in_dist, dist_in_com.len(), 
                                mean_out_dist, dist_out_com.len());
    //
    (mean_in_dist, mean_out_dist)
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
    let communities = read_orkut_com(Path::new(ORKUT_DATA_DIR)).unwrap();
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
    let num = 0usize;
    log::info!("analyze_community num : {num}");
    let (d_in, d_out) = analyze_community(&communities[num], &orkut_graph, &orkut_embedding, 5000);

    let num: usize  = 1;
    log::info!("analyze_community num : {num}");    
    let (d_in, d_out) = analyze_community(&communities[num], &orkut_graph, &orkut_embedding, 5000);

    let num: usize  = 4;
    log::info!("analyze_community num : {num}");     
    let (d_in, d_out) = analyze_community(&communities[num], &orkut_graph, &orkut_embedding, 5000);

    let num: usize  = 22;
    log::info!("analyze_community num : {num}");   
    let (d_in, d_out) = analyze_community(&communities[num], &orkut_graph, &orkut_embedding, 5000);

} // end of main