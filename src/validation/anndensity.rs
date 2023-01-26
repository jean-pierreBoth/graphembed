//!  The purpose of this module is to evalate the embedding with structural properties conservation
//! Construct ann on embedding and compares with density decomposition of original graph

// We construct an Hnsw structure on embedded data.
// For each node of densest block we compute the fraction of its neighbour in the same block.
// We can compute density of each block and check if they are in the stable decomposition order



use std::time::{SystemTime};
use cpu_time::ProcessTime;

use anyhow::*;

use rayon::iter::{ParallelIterator, IntoParallelIterator};

use petgraph::graph::{Graph};
use petgraph::Undirected;

use ndarray::{Array2};

use hnsw_rs::prelude::*;
use hnsw_rs::flatten::FlatNeighborhood;



use crate::embedding::*;
use crate::structure::density::*;
use crate::structure::density::stable::*;


/// gathers distance statistics from a point to its neighbours
#[derive(Default, Debug, Copy,Clone)]
pub struct DistStat {
    /// block for which we collect stats
    blocnum : usize,
    /// mean distance among all neighbours reference in flat neighborhood
    mean_dist : f64,
    /// minimal dist of a neighbour in same block as point
    min_dist_in_block : f64,
}

impl DistStat {
    pub fn new(blocnum : usize, mean_dist : f64, min_dist_in_block : f64) -> Self {
        DistStat{blocnum , mean_dist, min_dist_in_block}
    }

    /// returns num of block for hich the structure maintains statistics
    pub fn get_blocnum(&self) -> usize  {
        self.blocnum
    }

    /// returns distance among all neighbours reference in flat neighborhood
    pub fn get_mean(&self) -> f64 {
        self.mean_dist
    }

    /// minimal dist among neighbour in same block as point
    pub fn get_min_dist_in_block(&self) -> f64 {
        self.min_dist_in_block
    }

} // end of DistStat





//====================================================================================================

/// builds the Hnsw from the embedded data
pub fn embeddedtohnsw<F, D>(embedded : & dyn EmbeddedT<F>, max_nb_connection : usize, ef_c : usize) -> Result<Hnsw<F, DistPtr<F, f64>>, anyhow::Error>
    where F : Copy+Clone+Send+Sync ,
          D : Distance<F> {
    //
    let distance_e = embedded.get_distance();
    let distance = DistPtr::<F, f64>::new(distance_e);
    let nbdata = embedded.get_nb_nodes();
    let nb_layer = 16;
    //
    let cpu_start = ProcessTime::now();
    let sys_start = SystemTime::now();
    //
    let hnsw = Hnsw::<F, DistPtr<F, f64>>::new(max_nb_connection, nbdata, 
                        nb_layer, ef_c, distance);
    // we will insert by blocks
    let block_size = 10000;
    let nb_blocks_min = nbdata / block_size;
    let nb_blocks;
    if nbdata > nb_blocks_min * block_size {
        nb_blocks = nb_blocks_min + 1;
    }
    else {
        nb_blocks = nb_blocks_min;
    }
    let mut nb_sent = 0;
    let mut embeded_v  = Vec::<(Vec<F>, usize)>::with_capacity(block_size);
    for block in 0..nb_blocks {
        embeded_v.clear();
        let first = block * block_size;
        let last = (first + block_size).min(nbdata);
        for rank in first..last {
            // TODO avoid this unnecessary copy!?
            let v = embedded.get_embedded_node(rank, TAG_IN_OUT).to_vec();
            embeded_v.push((v,rank));
        }
        let data_with_id : Vec::<(&[F], usize)> = embeded_v.iter().map(| data| (data.0.as_slice(),data.1)).collect();
        hnsw.parallel_insert_slice(&data_with_id);
        nb_sent += data_with_id.len();
    } 
    //
    let sys_t : f64 = sys_start.elapsed().unwrap().as_millis() as f64 / 1000.;
    println!(" embedding sys time(s) {:.2e} cpu time(s) {:.2e}", sys_t, cpu_start.elapsed().as_secs());
    //
    log::debug!("embedtohnsw , sent {nb_sent} to hnsw");
    //
    return Ok(hnsw);
} // end of embedtohnsw


// Caution : Must check coherence of node indexation between graph loading and csr matrix loading 
// but we use the same loading methods in crate::io::csv
/// TODO have many/better density 
fn compare_block_density(flathnsw : &FlatNeighborhood, stable : &StableDecomposition, blocnum : usize) -> (DistStat, Vec<f32>) {
    //
    log::info!("compare_block_density for block : {blocnum})");
    //
    let highest = stable.get_block_points(blocnum).unwrap();
    let mean_block_size = stable.get_mean_block_size();
    let block_size_out = highest.len();
    let nb_blocks = stable.get_nb_blocks();
    let mut block_counts = (0..nb_blocks).into_iter().map(|_| 0.).collect::<Vec<f32>>();
    //
    let mut min_dist_in_block = f64::INFINITY;
    // loop on points of highest block
    let mut density = 0.;
    let mut nb_non_zero_dist : usize = 0;
    for node in &highest {
        // get neighbours of node
        let neighbours = flathnsw.get_neighbours(*node).unwrap();
        for neighbour in &neighbours[0..neighbours.len().min(64)] {
            // search which block is neighbour
            let dist = neighbour.get_distance();
            let neighbour_blocknum = stable.get_densest_block(neighbour.get_origin_id()).unwrap();
            let block_size_in = stable.get_nbpoints_in_block(neighbour_blocknum).unwrap();
            if dist > 0. {
                nb_non_zero_dist += 1;
                if neighbour_blocknum == blocnum {
                    min_dist_in_block = min_dist_in_block.min(dist as f64);
                }
                density += dist;
            }
            // estimated density around node
            block_counts[neighbour_blocknum] += mean_block_size as f32/ ((block_size_in * block_size_out) as f32);
        }
    } // end of loop on node
    //
    let global_density = density as f64 / (nb_non_zero_dist as f64);
    log::info!("mean distance of neigbours in bloc {blocnum}: {:.3e}", global_density);
    let diststat = DistStat::new(blocnum, global_density, min_dist_in_block);
    //
    return (diststat,block_counts);
}  // end of compare_block_density




/// get fraction of edge out of block. recall That B_{i} = \cup S_{j} for j <= i
fn get_block_stats(blocnum : usize, blockout : &Vec<f32>) -> f64 {

    let mut out = 0.;
    let mut nb_edges = 0.;
    for j in 0..blockout.len() {
        nb_edges += blockout[j];
        if j > blocnum {
            out += blockout[j];
        }
    }
    //
    let frac_out : f64 = out as f64 / nb_edges as f64;
    //
    frac_out
} // end of get_block_stats



/// A tentative assesment of embedding by density comparison before and after embedding
/// For each block *b* of density decomposition we analyze in which  blocks are the neigbours of points in b.
/// For a block *b* of high density we expect its neighbours to be significantly also in *b*.
/// This can be assessed by computing transition probability between blocks along edges.
pub fn density_analysis<F,D, N>(graph : &Graph<N, f64, Undirected>, embedded : &Embedded<F>, 
            hnsw_opt : Option<Hnsw<F,DistPtr<F,f64>>>, 
            decomposition_opt : Option<StableDecomposition>) -> Result<Array2<f32>, anyhow::Error>
    where F : Copy+Clone+Send+Sync ,
          D : Distance<F> , 
          N : std::marker::Copy  {
    // temporarily we pass stable decomposition as optional arg
    let decomposition = match decomposition_opt {
        Some(decomposition) => {decomposition},
        None => {
            let nb_iter = 500;
            approximate_decomposition(&graph, nb_iter)  
        }
    };

    let hnsw = match hnsw_opt {
        Some(hnsw) => {hnsw},
        None => {
            // TODO : adapt parameters to decomposition result
            let max_nb_connection : usize = decomposition.get_mean_block_size().min(64);
            log::info!("density_analysis : using max_nb_onnection : {max_nb_connection}");
            let ef_construction : usize = 48;
            let hnsw_res  = embeddedtohnsw::<F,D>(embedded, max_nb_connection, ef_construction);
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
    let nb_dense_blocks = nb_blocks.min(250);
    let res_analysis : Vec<(DistStat, Vec<f32>)> = (0..nb_dense_blocks).into_par_iter().
            map(|i| compare_block_density(&flathnsw, &decomposition, i)).collect();
    //
    // make a matrix of transitions between blocks
    //
    let block_transition = Array2::<f32>::from_shape_fn((nb_blocks, nb_blocks),
                    |(i,j)| res_analysis[i].1[j]
                );
    //
    log::info!("\n\n nb neighbours by blocks");
    for (d, v) in &res_analysis {
        log::info!("\n\n density_analysis for densest block : {:#?}, block size : {:#?}", d, decomposition.get_nbpoints_in_block(d.get_blocnum()).unwrap());
        let frac_out = get_block_stats(d.get_blocnum(), v);
        log::info!(" fraction out : {:.2e}", frac_out);
        log::info!("block : {:?}", v);
    } 
    //
    return Ok(block_transition);
} // end of density_analysis

//========================================================================================================


#[cfg(test)] 
mod tests {
   
    use super::*;

    use crate::prelude::*;
    use petgraph::{Undirected};

    use crate::io::csv::weighted_csv_to_graphmap;
    use crate::io::csv::csv_to_trimat;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    } 

    #[test]
    fn ann_check_density_miserables() {
        //
        log_init_test();
        //
        log::debug!("in anndensity ann_check_density_miserables");
        let path = std::path::Path::new(crate::DATADIR).join("moreno_lesmis").join("out.moreno_lesmis_lesmis");
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
        let params = NodeSketchParams{sketch_size, decay, nb_iter, symetric, parallel};
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
        let block_analysis = density_analysis::<usize, DistPtr<usize, f64>, u32>(&graph, &embedded, None, None);
        if block_analysis.is_err() {
            log::error!("block_analysis failed");
        }
        //
        // we transform into hnsw
        let _max_nb_connection = 24usize;
        let _ef_construction = 48usize;
        
    } // end of ann_check_density_miserables

}  // end of mod test