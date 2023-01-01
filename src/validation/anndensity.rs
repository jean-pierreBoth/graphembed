//!  The purpose of this module is to evalate the embedding with structurzl properties conservation
//! Construct ann on embedding and compares with density decomposition of original graph

// We construct an Hnsw structure on embedded data.
// For each node of densest block we the fraction of its neighbour in the same block.
// We can compute density of each block and check if they are in the stable decomposition order

#![allow(unused)]

use std::time::{SystemTime};
use cpu_time::ProcessTime;

use anyhow::*;

use num_traits::{float::*, FromPrimitive};

// synchronization primitive
use std::sync::{Arc};
use parking_lot::{RwLock};
use rayon::prelude::*;


use petgraph::graph::{Graph, EdgeReference, NodeIndex};
use petgraph::{Undirected, visit::*};


use hnsw_rs::prelude::*;
use hnsw_rs::flatten::FlatNeighborhood;

use annembed::prelude::*;


use crate::structure::*;
use crate::embedding::*;
use crate::structure::density::stable::*;


/// builds the Hnsw from the embedded data
fn embeddedtohnsw<F, D>(embedded : &Embedded<F>) -> Result<Hnsw<F, DistPtr<F, f64>>, anyhow::Error>
    where F : Float+Clone+Send+Sync ,
          D : Distance<F> {
    //
    let distance_e = embedded.get_distance();
    let distance = DistPtr::<F, f64>::new(distance_e);
    let ef_c = 50;
    let max_nb_connection = 64;
    let nbdata = embedded.get_nb_nodes();
    let nb_layer = 16.min((nbdata as f32).ln().trunc() as usize);
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
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
    let mut rank = 0;
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
    log::debug!("embedtohnsw , sent {nb_sent} to hnsw");
    //
    return Ok(hnsw);
} // end of embedtohnsw



// Caution : Must check coherence of node indexation between graph loading and csr matrix loading 
// but we use the same loading methods in crate::io::csv
fn compare_density<F,D>(hnsw : &Hnsw<F, DistPtr<F, f64>>, stable : &StableDecomposition) 
    where F : Float+Clone+Send+Sync ,
    D : Distance<F>  {
    //
    let flathnsw = FlatNeighborhood::from(hnsw);
    //
    let blocnum = 0;
    let highest =  stable.get_block_points(blocnum).unwrap();
    //
    // loop on points of highest block
    let mut neighbours_same_block = 0;
    let mut nb_neighbours = 0;
    let mut density = 0.;
    for node in &highest {
        // get neighbours of node
        let neighbours = flathnsw.get_neighbours(*node).unwrap();
        for neighbour in &neighbours {
            // search which block is neighbour
            let neighbour_blocknum = stable.get_densest_block(neighbour.get_origin_id()).unwrap();
            // estimated density around node
            if neighbour_blocknum == blocnum {
                neighbours_same_block += 1;
            }
            density += neighbour.get_distance();
        }
        nb_neighbours += neighbours.len();
    } // end of loop on node
    //
    let same_block_fraction = neighbours_same_block as f32 / nb_neighbours as f32;
    let global_density = density / ((nb_neighbours * highest.len()) as f32);
    log::info!("fraction neigbours of same block : {:.3e}", same_block_fraction);
    log::info!("mean distance to neigbours : {:.3e}", global_density);

}  // end of compare_density


#[cfg(test)] 
mod tests {
   
    use super::*;

    
    use crate::io::csv::weighted_csv_to_graphmap;
    use crate::structure::density::pava::{PointIterator};
    use crate::io::csv::csv_to_trimat;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    } 

    #[test]
    fn ann_check_pava_miserables() {
        log::debug!("in anndensity ann_check_pava_miserables");
        let path = std::path::Path::new(crate::DATADIR).join("moreno_lesmis").join("out.moreno_lesmis_lesmis");
        log::info!("\n\n algodens::density_miserables, loading file {:?}", path);
        let res = csv_to_trimat::<f64>(&path, false, b' ');
        if res.is_err() {
            log::error!("algodens::density_miserables failed in csv_to_trimat");
            assert_eq!(1, 0);
        }
    } // end of ann_check_pava_miserables

}  // end of mod test