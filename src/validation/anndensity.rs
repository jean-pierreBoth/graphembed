//!  The purpose of this module is to evalate the embedding with structural properties conservation.  
//!  We analyze how distances inside blocks of the stable decomposiiton behave after embedding.
//!
//! We construct ann neighbour graph on embedded data and analyze the distance between neighbours.
//! In particular for each node we consider its *nbmax* neighbours where *nbmax* is the min of its degree in the original graph
//! and the number of neighbours the ann has.  
//!  
//! We then compute the mean distance between the nodes when neighbour is inside the same block
//! and outside the block.
//! We also compute the distribution of blocks of neighbours and compare it with the distribution in the original graph with
//! Kullbach-Leibler divergence.
//!
//! The interface to the validation is the function [density_analysis] that takes as arguments the original graph and the embedded data.
//! Optional arguments can be specified to gain control over the Ann and densitiy decomposition.

// We construct an Hnsw structure on embedded data.
// For each node of densest block we compute the fraction of its neighbour in the same block.
// We can compute density of each block and check if they are in the stable decomposition order

use cpu_time::ProcessTime;
use std::time::SystemTime;

use anyhow::*;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use std::fs::OpenOptions;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::to_writer;

use petgraph::graph::Graph;
use petgraph::Undirected;

use hdrhistogram::Histogram;

use hnsw_rs::flatten::FlatNeighborhood;
use hnsw_rs::prelude::*;

use crate::embedding::*;
use crate::structure::density::stable::*;
use crate::structure::density::*;

/// Gathers statistics for each block obtained from the Ann flathnsw representation for comparison with those obtained of the
/// original graph representation.   
/// We collect mean distances inside a block and mean distance for edge crossing a block boundary.
/// We also collect transition probabilities between blocks.  
///
/// The data collected depend on parameter ef_construction used in Hnsw creation.
/// The higher ef_construction, the more representative are collected data, at the expense of cpu.  
/// ef = 48 or 64 seem a good compromise, a rule of thumb is ef should be higher than mean graph degree.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct BlockStat {
    /// block for which we collect stats
    blocnum: usize,
    /// total number of edge involving a node in this block
    block_degree: u32,
    /// mean distance among all neighbours reference in flat neighborhood.
    mean_dist: f64,
    ///  mean distance inside block in ann flat neighbourhood
    mean_dist_in: f32,
    /// mean dist of edge crossing block boundary in ann flat neighbourhood
    mean_dist_out: f32,
    /// minimal dist of a neighbour in same block as point
    min_dist_in_block: f64,
    /// transition probability
    transition_proba: Vec<f32>,
    /// kl divergence with original graph transitions
    kl_divergence: f32,
} // end of BlockStat

impl BlockStat {
    pub fn new(
        blocnum: usize,
        block_degree: u32,
        mean_dist: f64,
        mean_dist_in: f32,
        mean_dist_out: f32,
        min_dist_in_block: f64,
        transition_proba: Vec<f32>,
        kl_divergence: f32,
    ) -> Self {
        BlockStat {
            blocnum,
            block_degree,
            mean_dist,
            mean_dist_in,
            mean_dist_out,
            min_dist_in_block,
            transition_proba,
            kl_divergence,
        }
    }

    /// returns num of block for hich the structure maintains statistics
    pub fn get_blocnum(&self) -> usize {
        self.blocnum
    }

    /// returns total number of edge involving a node in this block
    pub fn get_block_degree(&self) -> u32 {
        self.block_degree
    }

    /// returns distance among all neighbours reference in flat neighborhood
    pub fn get_mean(&self) -> f64 {
        self.mean_dist
    }

    /// returns mean of distance between embedded pairs of point inside block
    pub fn get_mean_dist_in(&self) -> f32 {
        self.mean_dist_in
    }

    /// returns mean of distance between embedded pairs of point one inside , the other outside block
    pub fn get_mean_dist_out(&self) -> f32 {
        self.mean_dist_out
    }

    /// minimal dist among neighbour in same block as point
    pub fn get_min_dist_in_block(&self) -> f64 {
        self.min_dist_in_block
    }

    /// returns fraction of transition going out of block (i.e going into block of highrer index)
    pub fn get_fraction_out(&self) -> f32 {
        self.transition_proba[1 + self.blocnum..]
            .iter()
            .sum::<f32>()
    }

    /// for a given block, edges going out of block have an block arrival distribution.
    /// This function return kl_divergene the transition observed in the original graph and the embedded graph.
    pub fn get_kl_divergence(&self) -> f32 {
        self.kl_divergence
    }
} // end of BlockStat

/// This sturcture collects BlockStat statistics for all blocks
#[derive(Debug, Serialize, Deserialize)]
pub struct BlockCheck {
    blocks: Vec<BlockStat>,
} // end of BlockCheck

impl BlockCheck {
    /// get blocks of statistics
    pub fn get_blockstat(&self) -> &Vec<BlockStat> {
        &self.blocks
    }

    pub fn get_block(&self, numblock: usize) -> Result<&BlockStat> {
        if numblock >= self.blocks.len() {
            Err(anyhow!("BlockCheck get_block bzd block num arg"))
        } else {
            Ok(&self.blocks[numblock])
        }
    } // end of get_block

    /// dump in json format StableDecomposition structure
    pub fn dump_json(&self, filepath: &Path) -> Result<()> {
        //
        log::info!("dumping BlockCheck in json file : {:?}", filepath);
        //
        let fileres = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(filepath);
        if fileres.is_err() {
            log::error!(
                "BlockCheck dump : dump could not open file {:?}",
                filepath.as_os_str()
            );
            println!(
                "BlockCheck dump: could not open file {:?}",
                filepath.as_os_str()
            );
            return Err(anyhow!("BlockCheck dump failed".to_string()));
        }
        //
        let mut writer = BufWriter::new(fileres.unwrap());
        to_writer(&mut writer, &self).unwrap();
        //
        Ok(())
    } // end of dump_json

    /// returns a stable decomposiiton from a json dump
    pub fn reload_json(filepath: &Path) -> Result<Self> {
        log::info!("in BlockCheck::reload_json");
        //
        let fileres = OpenOptions::new().read(true).open(filepath);
        if fileres.is_err() {
            log::error!(
                "BlockCheck::reload_json : reload could not open file {:?}",
                filepath.as_os_str()
            );
            println!(
                "BlockCheck::reload_json: could not open file {:?}",
                filepath.as_os_str()
            );
            return Err(anyhow!(
                "BlockCheck::reload_json:  could not open file".to_string()
            ));
        }
        //
        let loadfile = fileres.unwrap();
        let reader = BufReader::new(loadfile);
        let blockcheck: Self = serde_json::from_reader(reader).unwrap();
        //
        log::info!("end of BlockCheck reload ");
        //
        Ok(blockcheck)
    } // end of reload_json

    /// get histogram of ratio dist_in / dist_out for blocks
    /// The lower is the median, the better is the result
    /// Dumps also the mean of ratio edge in / edge out weighted by block degree
    pub fn get_in_out_distance_ratio(&self) -> Vec<(f64, f64)> {
        let mut histo = Histogram::<u64>::new(2).unwrap();
        let scale: f32 = 500.;
        log::info!("analyzing edge in/ edge out in blocks");
        log::debug!("scaling ratios at {scale}");
        //
        self.blocks
            .iter()
            .for_each(|b| histo += (scale * b.get_mean_dist_in() / b.get_mean_dist_out()) as u64);
        let quantiles = vec![0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95];
        log::info!("quantiles used : {:?}", &quantiles);
        for q in &quantiles {
            log::info!(
                "value at q : {:.3e} = {:.3e}",
                q,
                histo.value_at_quantile(*q) as f64 / scale as f64
            );
        }
        //
        let mut nb_edges: usize = 0;
        let mut weighted_mean_ratio: f32 = 0.;
        for i in 0..self.blocks.len() - 1 {
            // we avoid last block as we know there is no edge out!
            let b = &self.blocks[i];
            nb_edges += b.get_block_degree() as usize;
            weighted_mean_ratio +=
                b.get_block_degree() as f32 * b.get_mean_dist_in() / b.get_mean_dist_out();
        }
        weighted_mean_ratio /= nb_edges as f32;
        log::info!(
            " blocks mean ratio edge in / edge out weighted by block degree : {:.3e}",
            weighted_mean_ratio
        );
        //
        let quant_res = quantiles
            .into_iter()
            .map(|q| (q, histo.value_at_quantile(q) as f64 / scale as f64))
            .collect::<Vec<(f64, f64)>>();
        for q in &quant_res {
            log::info!(
                " blocks quantiles ratio (edge in/ edge out) at proba {:.3e} = {:.3e}",
                q.0,
                q.1
            );
        }
        quant_res
    } // end of get_in_out_distance_ratio

    /// For a given block, edges having one node in a block have a neighbour in some arrival block (possibly the same block).
    /// We record this arrival distribution for each block and compute the kl_divergene between transitions
    /// observed in the original graph and the embedded graph.  
    /// This function returns histogram of kl_divergene between transitions observed when varying the departure block.
    pub fn get_divergence_histogram(&self) -> Vec<(f64, f64)> {
        log::info!("analyzing edge block transitions");

        let mut histo = Histogram::<u64>::new(2).unwrap();
        let scale: f32 = 500.;
        self.blocks
            .iter()
            .for_each(|b| histo += (scale * b.get_kl_divergence()) as u64);
        let quantiles = vec![0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95];
        log::info!("quantiles used : {:?}", &quantiles);
        for q in &quantiles {
            log::info!(
                "value at q : {:.3e} = {:.3e}",
                q,
                histo.value_at_quantile(*q) as f64 / scale as f64
            );
        }
        //
        let quant_res = quantiles
            .into_iter()
            .map(|q| (q, histo.value_at_quantile(q) as f64 / scale as f64))
            .collect::<Vec<(f64, f64)>>();
        for q in &quant_res {
            log::info!(" kl_divergence quantiles ratio between block transition law at proba {:.3e} = {:.3e}", q.0, q.1);
        }
        quant_res
    } // end of get_kl_divergence_histogram
} // end of impl BlockCheck

//====================================================================================================

/// Builds the Hnsw structure from the embedded data
/// In the Hnsw structure original nodes of the graph are identified by their NodeIndex or rank in embedded structure.  
/// (The N type of the graph structure is not used anymore at this step)
pub fn embeddedtohnsw<'a, 'b, F, D>(
    embedded: &'a dyn EmbeddedT<F>,
    max_nb_connection: usize,
    ef_c: usize,
) -> Result<Hnsw<'b, F, DistPtr<F, f64>>, anyhow::Error>
where
    F: Copy + Clone + Send + Sync,
    D: Distance<F>,
    'b: 'a,
{
    //
    let distance_e = embedded.get_distance();
    let distance = DistPtr::<F, f64>::new(distance_e);
    let nbdata = embedded.get_nb_nodes();
    let nb_layer = 16;
    //
    let cpu_start = ProcessTime::now();
    let sys_start = SystemTime::now();
    //
    let hnsw = Hnsw::<F, DistPtr<F, f64>>::new(max_nb_connection, nbdata, nb_layer, ef_c, distance);
    // we will insert by blocks
    let block_size = 10000;
    let nb_blocks_min = nbdata / block_size;
    let nb_blocks = if nbdata > nb_blocks_min * block_size {
        nb_blocks_min + 1
    } else {
        nb_blocks_min
    };
    let mut nb_sent = 0;
    let mut embeded_v = Vec::<(Vec<F>, usize)>::with_capacity(block_size);
    for block in 0..nb_blocks {
        embeded_v.clear();
        let first = block * block_size;
        let last = (first + block_size).min(nbdata);
        for rank in first..last {
            // TODO avoid this unnecessary copy!?
            let v = embedded.get_embedded_node(rank, TAG_IN_OUT).to_vec();
            embeded_v.push((v, rank));
        }
        let data_with_id: Vec<(&[F], usize)> = embeded_v
            .iter()
            .map(|data| (data.0.as_slice(), data.1))
            .collect();
        hnsw.parallel_insert_slice(&data_with_id);
        nb_sent += data_with_id.len();
    }
    //
    let sys_t: f64 = sys_start.elapsed().unwrap().as_millis() as f64 / 1000.;
    println!(
        " embedding sys time(s) {:.2e} cpu time(s) {:.2e}",
        sys_t,
        cpu_start.elapsed().as_secs()
    );
    //
    log::debug!("embedtohnsw , sent {nb_sent} to hnsw");
    //
    Ok(hnsw)
} // end of embedtohnsw

// We compute transition probabilities between blocks after embedding and compare it with data
// before embedding by computing K.L divergence between distributions for corresponding block
/// TODO have many/better density
fn compare_block_density(
    flathnsw: &FlatNeighborhood,
    stable: &StableDecomposition,
    blocnum: usize,
) -> BlockStat {
    //
    log::info!("compare_block_density for block : {blocnum}");
    //
    let highest = stable.get_block_points(blocnum).unwrap();
    let block_size_out = highest.len();
    let nb_blocks = stable.get_nb_blocks();
    let mut block_counts = (0..nb_blocks).map(|_| 0.).collect::<Vec<f32>>();
    //
    let mut min_dist_in_block = f64::INFINITY;
    // loop on points of highest block
    let mut density = 0.;
    let mut _nb_non_zero_dist: usize = 0;
    let mut nb_dist = 0;
    // to compare mean dist in block and crossing blck frontier
    let mut nb_dist_in = 0;
    let mut nb_dist_out = 0;
    let mut mean_dist_in = 0.;
    let mut mean_dist_out = 0.;
    for node in &highest {
        // get neighbours of node. The variable node here refers to the original NodeIndex.
        // No more N (weight in petgraph terminology)
        let neighbours = flathnsw.get_neighbours(*node).unwrap();
        let max_nb_nbg = stable.get_node_degree(*node);
        assert!(max_nb_nbg > 0);
        // we use as many neighbour we have in flathnsw but no more than degree of node in original graph.
        // Keeping track of degree is necessary to get a good profile of leakage out of blocks
        for neighbour in &neighbours[0..neighbours.len().min(max_nb_nbg)] {
            // search which block is neighbour
            let dist = neighbour.get_distance();
            let neighbour_blocknum = stable.get_densest_block(neighbour.get_origin_id()).unwrap();
            if dist > 0. {
                _nb_non_zero_dist += 1;
                if neighbour_blocknum == blocnum {
                    min_dist_in_block = min_dist_in_block.min(dist as f64);
                }
            }
            density += dist;
            nb_dist += 1;
            if neighbour_blocknum <= blocnum {
                nb_dist_in += 1;
                mean_dist_in += dist;
            } else {
                nb_dist_out += 1;
                mean_dist_out += dist;
            }
            // estimated density around node
            block_counts[neighbour_blocknum] += 1.0 / (block_size_out as f32);
        }
    } // end of loop on node
      // renormalize
    if nb_dist_in > 0 {
        mean_dist_in /= nb_dist_in as f32;
    }
    if nb_dist_out > 0 {
        mean_dist_out /= nb_dist_out as f32;
    }
    let sum = block_counts.iter().sum::<f32>();
    block_counts.iter_mut().for_each(|v| *v /= sum);
    //
    let global_density = density as f64 / (nb_dist as f64);

    // we can compute divergence between block_counts and corresponding measure in stable distribution
    let divergence = kl_divergence(
        &block_counts,
        stable
            .get_block_transition()
            .row(blocnum)
            .as_slice()
            .unwrap(),
    );
    log::info!(
        "mean distance of neigbours in bloc {blocnum}: {:.3e}, block divergence : {:.3e}",
        global_density,
        divergence
    );
    //
    let diststat = BlockStat::new(
        blocnum,
        nb_dist_in + nb_dist_out,
        global_density,
        mean_dist_in,
        mean_dist_out,
        min_dist_in_block,
        block_counts,
        divergence,
    );
    //
    diststat
} // end of compare_block_density

// computes kl divergence between 2 row of block transition. Array are normalized to 1.
fn kl_divergence(p1: &[f32], p2: &[f32]) -> f32 {
    let div = p1.iter().zip(p2.iter()).fold(0., |acc, v| {
        if v.0 > &0. {
            acc + v.0 * (v.1 / v.0).ln()
        } else {
            acc
        }
    });
    -div
} // end of kl_divergence

/// get fraction of edge out of block. recall That B_{i} = \cup S_{j} for j <= i
#[allow(unused)]
fn get_block_stats(blocnum: usize, blockout: &[f32]) -> f64 {
    let mut out = 0.;
    let mut nb_edges = 0.;
    for j in 0..blockout.len() {
        nb_edges += blockout[j];
        if j > blocnum {
            out += blockout[j];
        }
    }
    //
    let frac_out: f64 = out as f64 / nb_edges as f64;
    //
    frac_out
} // end of get_block_stats

/// A tentative assesment of embedding by density comparison edges length after embedding.  
///
/// The function computes an embedding and a decomposition in stable blocks. Then we compare
/// edge length of embedded data and check that distances betwwen nodes inside a same block and related by an edge
/// are smaller than distances between nodes related by an edge crossing a block frontier.
///
/// The user can provide the embedding and the stable decomposition or let the function compute them
/// by passing None args. (See function [embeddedtohnsw] to gain control over the ann computations)
///
/// Computing transition probabilities between blocks along edges and compared with the original graph.
///
pub fn density_analysis<F, D, N>(
    graph: &Graph<N, f64, Undirected>,
    embedded: &Embedded<F>,
    hnsw_opt: Option<Hnsw<F, DistPtr<F, f64>>>,
    decomposition_opt: Option<StableDecomposition>,
) -> Result<BlockCheck, anyhow::Error>
where
    F: Copy + Clone + Send + Sync,
    D: Distance<F>,
    N: std::marker::Copy,
{
    // temporarily we pass stable decomposition as optional arg
    let decomposition = match decomposition_opt {
        Some(decomposition) => decomposition,
        None => {
            let nb_iter = 500;
            log::info!("doing approximate_decomposition");
            approximate_decomposition(graph, nb_iter)
        }
    };

    let hnsw = match hnsw_opt {
        Some(hnsw) => hnsw,
        None => {
            // TODO : adapt parameters to decomposition result
            let max_nb_connection: usize = decomposition.get_mean_block_size().min(64);
            log::info!(
                "density_analysis : construction hnsw using max_nb_onnection : {max_nb_connection}"
            );
            let ef_construction: usize = 48;
            let hnsw_res = embeddedtohnsw::<F, D>(embedded, max_nb_connection, ef_construction);
            if hnsw_res.is_err() {
                return Err(anyhow!("density_analysis cannot do the hnsw construction"));
            }
            hnsw_res.unwrap()
        }
    };
    //
    let flathnsw = FlatNeighborhood::from(&hnsw);
    //
    let nb_blocks = decomposition.get_nb_blocks();
    // now we can loop ( //) on densest blocks and more populated blocks.
    let nb_max_blocks = 500;
    log::info!("keeping max {nb_max_blocks}");
    let nb_dense_blocks = nb_blocks.min(nb_max_blocks);
    let res_analysis: Vec<BlockStat> = (0..nb_dense_blocks)
        .into_par_iter()
        .map(|i| compare_block_density(&flathnsw, &decomposition, i))
        .collect();
    //
    log::info!("\n\n nb neighbours by blocks");
    for d in &res_analysis {
        log::info!(
            "\n\n density_analysis for densest block : {:?}, block size : {:#?}",
            d,
            decomposition
                .get_nbpoints_in_block(d.get_blocnum())
                .unwrap()
        );
        let frac_out = d.get_fraction_out();
        log::info!(
            " block : {:?}, fraction out : {:.2e}",
            d.get_blocnum(),
            frac_out
        );
    }
    //
    let blockcheck = BlockCheck {
        blocks: res_analysis,
    };
    log::info!("\n\n computing in out ratio for blocks");
    blockcheck.get_in_out_distance_ratio();
    //
    blockcheck.get_divergence_histogram();
    //
    Ok(blockcheck)
} // end of density_analysis

//========================================================================================================

#[cfg(test)]
mod tests {

    use super::*;

    use crate::prelude::*;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn ann_check_density_miserables() {
        //
        log_init_test();
        //
        log::debug!("in anndensity ann_check_density_miserables");
        let path = std::path::Path::new(crate::DATADIR)
            .join("moreno_lesmis")
            .join("out.moreno_lesmis_lesmis");
        log::info!("\n\n algodens::density_miserables, loading file {:?}", path);
        let res = csv_to_trimat::<f64>(&path, false, b' ');
        if res.is_err() {
            log::error!("algodens::density_miserables failed in csv_to_trimat");
            assert_eq!(1, 0);
        }
        let (trimat, node_index) = res.unwrap();
        // we embed
        let sketch_size = 20;
        let decay = 0.15;
        let nb_iter = 3;
        let parallel = false;
        let symetric = true;
        let params = NodeSketchParams {
            sketch_size,
            decay,
            nb_iter,
            symetric,
            parallel,
        };
        // now we embed
        let mut nodesketch = NodeSketch::new(params, trimat);

        let embedding_res = Embedding::new(node_index, &mut nodesketch);
        if embedding_res.is_err() {
            log::error!("test_nodesketch_lesmiserables failed in compute_Embedded");
            assert_eq!(1, 0);
        }
        let embedding = embedding_res.unwrap();
        let embedded = embedding.get_embedded_data();

        // we compute stable decomposition of les miserables
        // TODO get a conversion from trimat to GraphMap to avoid rereading!!
        let res = weighted_csv_to_graphmap::<u32, f64, Undirected>(&path, b' ');
        if res.is_err() {
            log::error!("algodens::density_miserables failed in csv_to_trimat");
            assert_eq!(1, 0);
        }
        // now we can convert into a Graph
        let graph = res.unwrap().into_graph::<u32>();
        //
        let block_analysis =
            density_analysis::<usize, DistPtr<usize, f64>, u32>(&graph, &embedded, None, None);
        if block_analysis.is_err() {
            log::error!("block_analysis failed");
        }
        //
        let block_check = block_analysis.unwrap();
        let _ratio_quants = block_check.get_in_out_distance_ratio();
    } // end of ann_check_density_miserables
} // end of mod test
