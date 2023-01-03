//!  The purpose of this module is to evalate the embedding with structural properties conservation
//! Construct ann on embedding and compares with density decomposition of original graph

// We construct an Hnsw structure on embedded data.
// For each node of densest block we the fraction of its neighbour in the same block.
// We can compute density of each block and check if they are in the stable decomposition order



use std::time::{SystemTime};
use cpu_time::ProcessTime;

use anyhow::*;

use rayon::iter::{ParallelIterator, IntoParallelIterator};

use petgraph::graph::{Graph};
use petgraph::Undirected;


use hnsw_rs::prelude::*;
use hnsw_rs::flatten::FlatNeighborhood;

//use annembed::prelude::*;


//use crate::structure::*;
use crate::embedding::*;
use crate::structure::density::*;
use crate::structure::density::stable::*;


/// builds the Hnsw from the embedded data
fn embeddedtohnsw<F, D>(embedded : &Embedded<F>, max_nb_connection : usize, ef_c : usize) -> Result<Hnsw<F, DistPtr<F, f64>>, anyhow::Error>
    where F : Copy+Clone+Send+Sync ,
          D : Distance<F> {
    //
    let distance_e = embedded.get_distance();
    let distance = DistPtr::<F, f64>::new(distance_e);
    let nbdata = embedded.get_nb_nodes();
    let nb_layer = 16.min((nbdata as f32).ln().trunc() as usize);
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
#[allow(unused)]
fn compare_block_density<F>(hnsw : &Hnsw<F, DistPtr<F, f64>>, stable : &StableDecomposition, blocnum : usize) -> Vec<usize>
    where F : Copy+Clone+Send+Sync {
    //
    log::info!("compare_block_density for block : {blocnum})");
    //
    let flathnsw = FlatNeighborhood::from(hnsw);
    //
    let highest =  stable.get_block_points(blocnum).unwrap();
    let nb_blocks = stable.get_nb_blocks();
    let mut block_counts = (0..nb_blocks).into_iter().map(|_| 0usize).collect::<Vec<usize>>();
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
            block_counts[neighbour_blocknum] += 1;
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
    log::info!("neighbour block counts : {:?}", block_counts);
    //
    return block_counts;
}  // end of compare_block_density



/// tentative assesment of embedding by density comparison before and after embedding
pub fn density_analysis<F,D, N>(graph : &Graph<N, f64, Undirected>, embedded : &Embedded<F>) -> Result<Vec<(usize, Vec<usize>)>, anyhow::Error>
    where F : Copy+Clone+Send+Sync ,
          D : Distance<F> , 
          N : std::marker::Copy  {
    //
    let nb_iter = 500;
    let decomposition = approximate_decomposition(&graph, nb_iter);    
    // TODO : adapt parameters to decomposition result
    let max_nb_connection : usize = 24;
    let ef_construction : usize = 48;
    let hnsw_res  = embeddedtohnsw::<F,D>(embedded, max_nb_connection, ef_construction);
    if hnsw_res.is_err() {
        return Err(anyhow!("density_analysis cannot do the hnsw construction"));    
    }
    let hnsw = hnsw_res.unwrap();
    //
    let nb_blocks = decomposition.get_nb_blocks();
    // now we can loop ( //) on densest blocks and more populated blocks.
    let nb_dense_blocks = nb_blocks.min(10);
    let res_analysis : Vec<(usize, Vec<usize>)> = (0..nb_dense_blocks).into_par_iter().
            map(|i| (i, compare_block_density(&hnsw, &decomposition, i))).collect();
    //
    log::info!("\n\n nb neighbours by blocks");
    for (i, v) in &res_analysis {
        log::info!("\n density_analysis for densest block : {i}");
        log::info!("block : {:?}", v);
    } 
    //
    return Ok(res_analysis);
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
        // we transform into hnsw
        let max_nb_connection = 24usize;
        let ef_construction = 48usize;

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
        let block_analysis = density_analysis::<usize, DistPtr<usize, f64>, u32>(&graph, &embedded);
        if block_analysis.is_err() {
            log::error!("block_analysis failed");
        }
        //
        /* 
        let nb_iter = 200;
        let res_hnsw = embeddedtohnsw::<usize, DistPtr<usize, f64>>(&embedded, max_nb_connection, ef_construction);
        let hnsw = res_hnsw.unwrap();
        let decomposition = approximate_decomposition(&graph, nb_iter);
        //
        let blocnum = 0;
        log::info!("\n\n density compariaison for blocnum : {blocnum}");
        compare_block_density(&hnsw, &decomposition, blocnum);

        let blocnum = 1;
        log::info!("\n\n density compariaison for blocnum : {blocnum}");
        compare_block_density(&hnsw, &decomposition, blocnum);

        let blocnum = 2;
        log::info!("\n\n density compariaison for blocnum : {blocnum}");
        compare_block_density(&hnsw, &decomposition, blocnum);

        let blocnum = 3;
        log::info!("\n\n density compariaison for blocnum : {blocnum}");
        compare_block_density(&hnsw, &decomposition, blocnum);

        let blocnum = 9;
        log::info!("\n\n density compariaison for blocnum : {blocnum}");
        compare_block_density(&hnsw, &decomposition, blocnum);
        */
    } // end of ann_check_density_miserables

}  // end of mod test