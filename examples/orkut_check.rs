//! example of density decompositon for Orkut graph [orkut](https://snap.stanford.edu/data/com-Orkut.html)
//! 
//! The graph is undirected with 3072441 nodes and 117 185 083 edges
//!
//!
//! The purpose of this example executable is to test coherence of orkut embedding.  
//! We analyze :
//!   - embedded distances inside communities and at community frontiers
//!   - embedded distance inside blocks of stable decomposition and at their frontier
//!
//! We reload embedding and  block decomposition dumped by orkut_hnsw example

/*

## community embedding ef_construction : usize = 48

### distances between nodes inside each community and their neighbours of which some can be outside community

for each community we compute the mean distance between 2 embedded neigbours inside community
and the mean distance between 2 neigbours when one is inside and the other out the community and
collect the statistics on the ratio

Histogram of ratio inside edge length / frontier crossing edge length :

[2023-01-27T14:46:31Z INFO  orkut_check] quantiles used : [0.05, 0.25, 0.5, 0.75, 0.95]
[2023-01-27T14:46:31Z INFO  orkut_check] value at q : 5.000e-2 = 2.010e2/500 = 0.4
[2023-01-27T14:46:31Z INFO  orkut_check] value at q : 2.500e-1 = 2.650e2/500 = 0.53
[2023-01-27T14:46:31Z INFO  orkut_check] value at q : 5.000e-1 = 3.130e2/500 = 0.626
[2023-01-27T14:46:31Z INFO  orkut_check] value at q : 7.500e-1 = 3.770e2/500 = 0.754
[2023-01-27T14:46:31Z INFO  orkut_check] value at q : 9.500e-1 = 5.470e2/500 = 1.094

We see that globally internal edges are embedded with smaller distances than edges crossing the community boundaries.
The ratio of the 2 mean length has the following properties:
- median at 0.62.
- for 5% of communities the ratio is greater than 1.1
- The mean ratio is 0.669
- The mean ratio, when weighted by community size is 0.697. This shows that there is no degradation of distances
  inside smaller communities, crossing community boundary is more penalized for small communities than large ones.

  But :
  Some very small communities for example 1493, 3719 and 4760 of respective sizes (3, 4 and 3)
  have internal edges more 2.5 larger than edges crossing boundary. 433 (among 5000) communities have mean internal edge greater than
  frontier crossing edges.

  ###  comparison of flow probability between blocks in the original graph and the ann graph (via Hnsw) computed from the embedding

  fraction out in original graph is 0.63 at first block ; increase up to 0.71 until block 40 then decrease to 0
  fraction out in embedded is 0.71 at first block increase up to 0.821 until block 40 then decrease

  KL divergence between original and embedded probability transitions between blocks begins at 0.029 originating from block 0 ,
  increase up to 0.82 for initial block 40, then decrease down to 0.058
  until block ~170 and then go up to 1. around last block last block.

  distances inside blocks are consistenty lower than distance crossing block boundary. 0.5 for first blocks up to 0.9 for lasts blocks.

  globally : embedded leaks out block a bit too much compared to original graph but follow the same pattern depending on blocks.

  TODO :
   - graphics, ef 64
  ####


*/

use anyhow::anyhow;

use std::fs::OpenOptions;
use std::io::prelude::*;
use std::path::Path;
use std::str::FromStr;

use std::io::BufReader;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use petgraph::prelude::*;
use petgraph::stable_graph::DefaultIx;

use graphembed::prelude::*;
use graphembed::validation::anndensity::*;

use hnsw_rs::hnswio::*;

use hnsw_rs::prelude::DistPtr;

use hdrhistogram::Histogram;

/// Directory containing the 2 data files
/// TODO: use clap in main
const ORKUT_DATA_DIR: &'static str = "/home/jpboth/Data/Graph/Orkut/";
const DUMP_DIR: &'static str = "/home/jpboth/Rust/graphembed/";

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

// read community file
fn read_orkut_com(dirpath: &Path) -> anyhow::Result<Vec<Vec<u32>>> {
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
    let mut communities = Vec::<Vec<u32>>::with_capacity(5000);
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
        let mut community: Vec<u32> = splitted
            .iter()
            .map(|s| u32::from_str(*s).unwrap())
            .collect();
        // we need to sort to use binary_search
        community.sort_unstable();
        assert!(community[0] != 0);
        communities.push(community);
        numline += 1;
    }
    //
    return Ok(communities);
} // end of read_orkut_com

fn jaccard_distance<T: Eq>(v1: &[T], v2: &[T]) -> f64 {
    assert_eq!(v1.len(), v2.len());
    let common = v1
        .iter()
        .zip(v2.iter())
        .fold(0usize, |acc, v| if v.0 == v.1 { acc + 1 } else { acc });
    1. - (common as f64) / (v1.len() as f64)
} // end of jaccard

//

// try to examine what happens to community through the embedding
// compares edges inside community with edges going out of community
// possibly : which edges in are matched in hnsw?
fn analyze_community(
    community: &Vec<u32>,
    graph: &Graph<u32, f64, Undirected>,
    orkut_embedding: &Embedding<usize, usize, Embedded<usize>>,
) -> (f64, f64) {
    // is_sorted is unstable
    // assert!(community.is_sorted());
    // loop , sample couple, compute distance
    let mut dist_in_com = Vec::<f64>::with_capacity(1000);
    let mut dist_out_com = Vec::<f64>::with_capacity(1000);
    //
    let node_indexation = get_graph_indexation(graph);
    //
    // loop with acceptance rejection on nodes.
    let com_size = community.len();

    for i in 0..com_size {
        // recall that community contains nodes names (type N in Graph<N,   >) and note indices!
        let node1 = community[i];
        assert!(node1 != 0); // beccause we know there is no node with name 0 in csv file!!
                             // must get the index corresponding to node which the node weight in petgraph terminology
        let n1_idx = node_indexation.get_index_of(&node1).unwrap();
        // iterate over neighbours of each node
        let mut neighbours = graph.neighbors_undirected(NodeIndex::new(n1_idx as usize));
        while let Some(n2_idx) = neighbours.next() {
            // now  n2_idx is a NodeInddex is neighbour n in community we must get its name
            let n2_name = graph[n2_idx];
            if n2_name == 0 {
                log::info!("n : {:?}", n2_idx);
                assert!(n2_name != 0);
            }
            let is_in = community.binary_search(&n2_name).is_ok();
            log::trace!("dist node1 : {:?} node2 : {:?}", node1, n2_name);
            let dist = orkut_embedding.get_node_distance(node1 as usize, n2_name as usize);
            if is_in {
                dist_in_com.push(dist);
            } else {
                dist_out_com.push(dist);
            }
        }
    }
    //
    let mean_in_dist = if dist_in_com.len() > 0 {
        dist_in_com.iter().sum::<f64>() / dist_in_com.len() as f64
    } else {
        0.
    };
    let mean_out_dist = dist_out_com.iter().sum::<f64>() / dist_out_com.len() as f64;
    //
    log::debug!("mean embedded distance between neighbours, in : {:.3e} len : {:?}, out : {:.3e} len : {:?}", mean_in_dist, dist_in_com.len(), 
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
    let dump_path = Path::new(DUMP_DIR);
    let fname = "orkut-decomposition.json";
    //
    let fileres = OpenOptions::new().read(true).open(&dump_path.join(fname));
    if fileres.is_err() {
        log::error!(
            "reload could not open file {:?}, will do decomposition",
            dump_path.join(fname)
        );
        log::error!(
            "orkut-decomposition.json not found, run orkut_hnsw to be able to reload decomposition"
        );
        panic!(
            "orkut-decomposition.json not found, run orkut_hnsw to be able to reload decomposition"
        );
    }
    log::info!("found json file for stored decomposition");
    // we reload decomposition
    let res = StableDecomposition::reload_json(&dump_path.join(fname));
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
    let orkut_embedding: Embedding<usize, usize, Embedded<usize>>;
    let orkut_bson_path = Path::new(DUMP_DIR).join("orkut_embedded.bson");
    let fileres = OpenOptions::new().read(true).open(&orkut_bson_path);
    if fileres.is_err() {
        log::info!(
            "cannot reoad orkut embedding, did not find file : {:?}",
            orkut_bson_path
        );
        std::panic!(
            "cannot reload orkut embedding, did not find file : {:?}",
            orkut_bson_path
        );
    } else {
        log::info!(
            "found bson file, reloading embedding, trying to reload from {:?}",
            &orkut_bson_path
        );
        let reloaded =
            bson_load::<usize, usize, Embedded<usize>>(orkut_bson_path.to_str().unwrap());
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
    let path = Path::new(DUMP_DIR);
    let reloader = HnswIo::new(path.to_path_buf(), String::from("orkuthnsw"));
    let mydist = DistPtr::<usize, f64>::new(jaccard_distance::<usize>);

    let hnsw_loaded = reloader.load_hnsw_with_dist(mydist);
    if hnsw_loaded.is_err() {
        log::error!("could not happen reload hnsw from path : {:?}", path);
        std::panic!();
    }
    let hnsw_loaded = hnsw_loaded.unwrap();
    //
    //  now we can check how are embedded blocks
    //
    let d_res = density_analysis::<usize, DistPtr<usize, f64>, DefaultIx>(
        &orkut_graph,
        orkut_embedding.get_embedded_data(),
        Some(hnsw_loaded),
        Some(decomposition),
    );
    if d_res.is_err() {
        log::error!(
            "density analysis failed with error: {:?}",
            d_res.as_ref().err()
        );
        std::process::exit(1);
    }

    log::info!("\n\n dumping density analysis result");
    let block_check = d_res.unwrap();
    let block_check_path = Path::new("orkut_block_check.json");
    block_check.dump_json(block_check_path).unwrap();
    //
    log::info!("exiting from orkut_check");
    //
    // now we can check how  communities are embedded we examined in Notebook
    //
    let ratios: Vec<f64> = (0..5000)
        .into_par_iter()
        .map(|num| {
            log::info!(
                "\n analyze_community num : {num}, size : {:?}",
                &communities[num].len()
            );
            let (d_in, d_out) =
                analyze_community(&communities[num], &orkut_graph, &orkut_embedding);
            d_in / d_out
        })
        .collect::<Vec<f64>>();

    let mut histo = Histogram::<u64>::new(2).unwrap();
    let scale = 500.;
    log::info!("scaling ratios at {scale}");
    ratios
        .iter()
        .for_each(|ratio| histo += (*ratio * scale) as u64);
    let quantiles = vec![0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95];
    log::info!("quantiles used : {:?}", quantiles);
    for q in quantiles {
        log::info!(
            "value at q : {:.3e} = {:.3e}",
            q,
            histo.value_at_quantile(q) as f64 / scale
        );
    }

    //
    let mut nb_greater = 0;
    let mut size_weighted_ratio: f64 = 0.;
    let mut unweighted_ratio: f64 = 0.;
    let mut size_weighted = 0usize;
    let mut sum_size_bad = 0usize;
    for i in 0..communities.len() {
        unweighted_ratio += ratios[i];
        size_weighted_ratio += ratios[i] * communities[i].len() as f64;
        size_weighted += communities[i].len();
        if ratios[i] >= 1. {
            log::info!(
                "community : {i}, size : {:?}, ratio : {:.3e}",
                communities[i].len(),
                ratios[i]
            );
            sum_size_bad += communities[i].len();
            nb_greater += 1;
        }
    }
    size_weighted_ratio /= size_weighted as f64;
    unweighted_ratio /= communities.len() as f64;
    let mean_size_bad = sum_size_bad as f64 / nb_greater as f64;
    log::info!(
        "mean community size : {:.3e}",
        size_weighted as f64 / communities.len() as f64
    );
    log::info!(
        "number of communites with greater in than out dist : {:?}, mean com size : {:.3e}",
        nb_greater,
        mean_size_bad
    );
    log::info!("mean ratio dist_in/dist_out : {:.3e}", unweighted_ratio);
    log::info!(
        "mean ratio dist_in/dist_out weighted community size: {:.3e}",
        size_weighted_ratio
    );
} // end of main
