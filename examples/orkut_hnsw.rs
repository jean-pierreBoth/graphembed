//! example of density decompositon for Orkut graph [https://snap.stanford.edu/data/com-Orkut.html]
//!
//! The graph is undirected with 3072441 nodes and 117 185 083 edges
//! The output is the ordered list of densest stable sets
//!
//! Frank-Wolfe 500 iterations run in 12.3mn. The whole computation need 13.4mn on a 8 cores 2-thread/core laptop with i7 intel proc
//!
//! The purpose of this example is to test block decomposition, embed the graph and transform the result into a Hnsw structure
//! and store the results for posterior treatment.
//!
//! The block decomposition, embedding and hnsw steps need 35mn on the same 8 cores 2-thread/core laptop with i7 intel proc

use anyhow::anyhow;

use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::str::FromStr;

use petgraph::prelude::*;

use graphembed::io::*;
use graphembed::prelude::*;
use graphembed::validation::anndensity::embeddedtohnsw;

use hnsw_rs::prelude::*;

/// Directory containing the 2 data files
/// TODO use clap in main
const ORKUT_DATA_DIR: &'static str = "/home/jpboth/Data/Graphs/Orkut/";
const DUMP_DIR: &'static str = "/home/jpboth/graphembed/Runs/";

/// Read graph (given as csv) and ground truth communities
fn read_orkut_graph(dirpath: &Path) -> Result<Graph<u32, f64, Undirected, u32>, anyhow::Error> {
    let fpath = dirpath.join("com-orkut.ungraph.txt");
    // use csv to unweighted graph_map
    log::info!("read_orkut_graph : reading {fpath:?}");
    let graphmap = weighted_csv_to_graphmap::<u32, f64, Undirected>(&fpath, b'\t');
    log::info!("read_orkutdir : reading {fpath:?}, done");
    if graphmap.is_err() {
        std::panic!("cannot open file : {fpath:?}");
    }
    let graph = graphmap.unwrap().into_graph::<u32>();
    log::info!("graph loaded");
    Ok(graph)
} // end of read_orkutgraph

/// returns a Vector of community. Each community is sorted
fn read_orkut_com(dirpath: &Path) -> anyhow::Result<Vec<Vec<usize>>> {
    let fpath = dirpath.join("com-orkut.top5000.cmty.txt");
    log::info!("read_orkut_com : reading {fpath:?}");
    let fileres = OpenOptions::new().read(true).open(&fpath);
    if fileres.is_err() {
        log::error!(
            "read_orkut_com : reload could not open file {:?}",
            fpath.as_os_str()
        );
        println!("read_orkut_com could not open file {:?}", fpath.as_os_str());
        return Err(anyhow!(
            "read_orkut_com could not open file {}",
            fpath.display()
        ));
    }
    let file = fileres?;
    let bufreader = BufReader::new(file);
    let lines = bufreader.lines();
    let mut numline = 0;
    let mut communities = Vec::<Vec<usize>>::with_capacity(5000);
    for line in lines {
        if line.is_err() {
            log::error!(
                "error reading file : {:?} at line : {}",
                fpath.as_os_str(),
                numline
            );
            return Err(anyhow!(
                " error reading file : {:?} at line : {}",
                fpath.as_os_str(),
                numline
            ));
        }
        // split and decode line. line consists in usize separated by a tab
        let line = line.unwrap();
        let splitted: Vec<&str> = line.split('\t').collect();
        let mut communitiy: Vec<usize> = splitted
            .iter()
            .map(|s| usize::from_str(*s).unwrap())
            .collect();
        // we need to sort indexes
        communitiy.sort_unstable();
        //
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
    log::info!("orkut graph read");
    //
    let orkut_graph = orkut_graph.unwrap();
    let decomposition: StableDecomposition;
    let nb_iter = 500;
    //
    // check if we have a stored decomposition
    //
    let dump_path = Path::new(&DUMP_DIR);
    //
    let fileres = OpenOptions::new()
        .read(true)
        .open(&dump_path.join("orkut-decomposition.json"));
    if fileres.is_err() {
        log::error!(
            "reload could not open file {:?}, will do decomposition",
            dump_path.as_os_str()
        );
        log::info!(
            " reload could not open file {:?}, will do decomposition ",
            dump_path.as_os_str()
        );
        println!(
            " reload could not open file {:?}, will do decomposition ",
            dump_path.as_os_str()
        );
        decomposition = approximate_decomposition(&orkut_graph, nb_iter);
        // and dump decomposition
        let res = decomposition.dump_json(&dump_path);
        match res {
            Ok(_) => {
                log::info!("orkut decomposition dumped in {dump_path:?} : Ok");
            }
            Err(_) => {
                log::info!("orkut decomposition dump failed in {dump_path:?} : Err");
            }
        };
        let nb_blocks = decomposition.get_nb_blocks();
        log::info!("orkut decomposition got nb_block : {nb_blocks}");
    } else {
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
    let orkut_embedding: Embedding<usize, usize, Embedded<usize>>;
    let fileres = OpenOptions::new()
        .read(true)
        .open(&dump_path.join("orkut_embedded.bson"));
    if fileres.is_err() {
        log::info!("going to embed orkut graph");
        let path = Path::new(ORKUT_DATA_DIR);
        // TODO avoid reloading. need a conversion trimat to graphmap
        let res = csv_to_trimat_delimiters::<f64>(&path.join("com-orkut.ungraph.txt"), false);
        let (orkut_trimat, node_index) = res.unwrap();
        let sketch_size = 200;
        let decay = 0.2;
        let nb_iter = 5;
        let parallel = true;
        let symetric = true;
        let params = NodeSketchParams {
            sketch_size,
            decay,
            nb_iter,
            symetric,
            parallel,
        };
        let mut nodesketch = NodeSketch::new(params, orkut_trimat);
        let sketch_embedding = Embedding::new(node_index, &mut nodesketch);
        if sketch_embedding.is_err() {
            log::error!("embedding orkut failed in Embedding::new");
            assert_eq!(1, 0);
        }
        orkut_embedding = sketch_embedding.unwrap();
        // now we can do a bson dump
        log::info!("dumping orkut embedding ...");
        let output = output::Output::new(
            output::Format::BSON,
            true,
            &Some(String::from("orkut_embedded")),
        );
        let bson_res = embeddedbson::bson_dump(&orkut_embedding, &output);
        if bson_res.is_err() {
            log::error!("bson dump in file {} failed", &output.get_output_name());
            log::error!("error returned : {:?}", bson_res.err().unwrap());
            assert_eq!(1, 0);
        }
        log::info!("orkut embedding done");
    } else {
        let reload_file = dump_path.join("orkut_embedded.bson");
        log::info!(
            "found bson file, reloading embedding, trying to reload from {:?}",
            &reload_file.to_str()
        );
        let reloaded =
            embeddedbson::bson_load::<usize, usize, Embedded<usize>>(reload_file.to_str().unwrap());
        if reloaded.is_err() {
            log::error!("reloading of bson from {:?} failed", &reload_file.to_str());
            log::error!("error is : {:?}", reloaded.err());
            std::panic!("bson reloading failed");
        }
        let bson_reloaded = reloaded.unwrap();
        let embedding = from_bson_with_jaccard(bson_reloaded);
        orkut_embedding = embedding.unwrap();
    }
    //
    //    std::process::exit(1);
    //
    // we compute hnsw on embedding
    //
    let max_nb_connection: usize = decomposition.get_mean_block_size().min(64);
    log::info!("hnsw construction using max_nb_onnection : {max_nb_connection}");
    let ef_construction: usize = 64;
    let hnsw_res = embeddedtohnsw::<usize, DistPtr<usize, f64>>(
        orkut_embedding.get_embedded_data(),
        max_nb_connection,
        ef_construction,
    );
    if hnsw_res.is_err() {
        log::error!(
            "embeddedtohnsw failed with error : {:?}",
            hnsw_res.as_ref().err()
        );
    }
    let hnsw = hnsw_res.unwrap();
    // some loggin info
    hnsw.dump_layer_info();
    // dump in a file. Must take care of name as tests runs in // !!!
    let fname = String::from("orkuthnsw");
    let _res = hnsw.file_dump(&fname);
    //
    log::info!("you can run orkut_check");
} // end of main
