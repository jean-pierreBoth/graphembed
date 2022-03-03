//! a simple link prediction to test Embeddeds

#![allow(unused)]


/// The scoring function for similarity is based upon the distance relative to the Embedded being tested
/// Jaccard for nodesketch and L2 for Atp
/// We first implement precision measure as described in 
///       - Link Prediction in complex Networks : A survey
///             Lü, Zhou. Physica 2011
///
///  Another reference is :
///       - Link prediction problem for social networks
///             David Liben-Nowell, J Kleinberg 2007

use log::*;
use anyhow::{anyhow};

use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;
use rand::distributions::{Uniform, Distribution};

use std::collections::{HashSet, HashMap};
use indexmap::IndexSet;

use sprs::{TriMatI, CsMatI};
use ndarray::{Array1, Array2};

use rayon::prelude::*;
use parking_lot::RwLock;

use crate::embedding::{Embedded, EmbeddedT, EmbedderT};
use crate::sketching::{nodesketch::*, nodesketchasym};
use crate::atp::*;

// filter out edge with proba delete_proba
// TODO currently avoid deleting diagonal terms (for Nodesketch) to parametrize
// The return a tuple containing the matrix of the graph with some edge deleted and a perfect hash on deleted edges
fn filter_csmat<F>(csrmat : &CsMatI<F, usize>, delete_proba : f64, symetric : bool, rng : &mut Xoshiro256PlusPlus) -> (TriMatI::<F, usize>, IndexSet<(usize,usize)>)
    where F: Copy {
    //
    assert!(delete_proba < 1. && delete_proba > 0.);
    log::trace!("filter_csmat delete_proba : {}", delete_proba);
    // get total in + out degree
    let mut degree = csrmat.degrees();
    let nb_nodes = csrmat.rows();
    let nb_edge = csrmat.nnz();
    let mut deleted_pairs = Vec::<(usize, usize)>::with_capacity((nb_edge as f64 * delete_proba).round() as usize);
    let mut deleted_edge = IndexSet::with_capacity((nb_edge as f64 * delete_proba).round() as usize);
    //
    log::info!("mean degree : {}", degree.iter().sum::<usize>() as f64/ nb_nodes as f64);
    //
    let uniform = Uniform::<f64>::new(0., 1.);
    let mut rows = Vec::<usize>::with_capacity(nb_nodes);
    let mut cols = Vec::<usize>::with_capacity(nb_nodes);
    let mut values = Vec::<F>::with_capacity(nb_nodes);
    //
    let mut not_discarded = 0usize;
    let mut csmat_iter = csrmat.iter();
    let mut discard = false;
    let mut discarded = 0usize;
    let nb_to_discard = ((nb_edge - nb_nodes) as f64 * delete_proba) as usize;
    // 
    log::debug!("csrmat nb edge : {}", nb_edge);
    // TODO must loop until we have deleted excatly the required number of edges
    while let Some((value,(row, col))) = csmat_iter.next() {
        let xsi = uniform.sample(rng);
        discard = false;
        if xsi < delete_proba && row < col {
            discard = true;
            // we must check we do not make an isolated node
            if degree[row] <= 1 || degree[col] <= 1 {
                not_discarded += 1;
                discard = false;
            }
        }
        if !discard {
            rows.push(row);
            cols.push(col);
            values.push(*value);
        }
        else {
            deleted_edge.insert((row, col));
            degree[row] -= 1;
            degree[col] -= 1;
            discarded += 1;
            if symetric {
                deleted_edge.insert((col, row));
                discarded += 1;
            }
        }
    }  // end while
    //
    log::info!(" ratio discarded = {:}", discarded as f64/ (nb_edge - nb_nodes) as f64);
    log::info!(" not discarded due to low degree {:?}", not_discarded);
    //
    let trimat = TriMatI::<F, usize>::from_triplets((nb_nodes,nb_nodes), rows, cols, values);
    //
    (trimat, deleted_edge)
} // end of filter_csmat



// local edge type corresponding to node1, ndde2 , distance from node1 to node2
#[derive(Copy, Clone, Debug)]
struct Edge(usize,usize,f64);






/// filters the full graph matrix edges with with rate delete_proba.
/// embed the filtered and sort all (seen and not seen) edges 
/// according to Embedded similarity function and return. 
/// ratio of really filtered out / over the number of deleted edges (i.e precision of the iteration)
/// type G is necessary beccause we embed in possibly different type than F. (for example in Array<usize> with nodesketch)
/// 
/// return (precision, recall)
fn one_precision_iteration<F, G, E>(csmat : &CsMatI<F, usize>, delete_proba : f64, symetric : bool, 
            embedder : &dyn Fn(TriMatI<F, usize>) -> E, rng : &mut Xoshiro256PlusPlus) -> (f64 , f64)
            where   F: Copy + std::marker::Sync ,
                    E : EmbeddedT<G> + std::marker::Sync {
    // filter
    let (mut trimat, deleted_edges) = filter_csmat(csmat, delete_proba, symetric, rng);
    // We construct the list of edges not in reduced graph
    let nb_nodes = csmat.shape().0;
    let max_degree = csmat.degrees().into_iter().max().unwrap();
    let mut embedded_edges = Vec::<Edge>::with_capacity(nb_nodes*nb_nodes);
    // filter out self loops and edges in trimat.
    let f_i = |i : usize| -> Vec<Edge> {
        let mut edges_i = Vec::<Edge>::with_capacity(max_degree);
        for j in 0..nb_nodes {
            if j!= i && (&trimat).find_locations(i,j).len() == 0usize {
                edges_i.push(Edge{0:i, 1:j, 2: 0f64});
            }
        }
        edges_i                       
    };
    let mut row_embedded : Vec<Vec<Edge>> = (0..nb_nodes).into_par_iter().map(|i| f_i(i)).collect();
    for i in 0..nb_nodes {
        log::trace!("row i : {}, row_embedded {:?}", i, row_embedded[i]);
        embedded_edges.append(&mut row_embedded[i]);
    }
    log::debug!(" nb remaining edges after random deletion : {}", embedded_edges.len());
    //
    // embed (to be passed as a closure)
    //
    let embedded = &embedder(trimat);
    // now we can compute all edge valuation (similarities betwen nodes pairs) and sort in increasing order
    // find how many deleted edges are in upper part of the sorted edges.
    for edge in &mut embedded_edges {
        edge.2 = embedded.get_noderank_distance(edge.0, edge.1);
    }
    embedded_edges.sort_unstable_by(|ea, eb| ea.2.partial_cmp(&eb.2).unwrap());
    log::debug!("nb embedded edges : {}, smallest : {:?}, largest : {:?}", embedded_edges.len(),embedded_edges[0], embedded_edges[embedded_edges.len()-1]);
    let retrieved = 80usize;
    embedded_edges.truncate(retrieved);
    log::debug!("\n smallest non diagonal edges remaining {:?} ... {:?}", embedded_edges[0], embedded_edges[embedded_edges.len()-1]);
    //
    let mut h_edges = HashMap::<(usize,usize), f64>::with_capacity(embedded_edges.len());
    for edge in embedded_edges {
        h_edges.insert((edge.0, edge.1), edge.2);
    }
    log::debug!("h_edges contains : {} edges ", h_edges.len());
    let mut nb_in = 0;
    for edge in &deleted_edges {
        if h_edges.contains_key(&(edge.0, edge.1)) {
            nb_in += 1;
        }
        if symetric && h_edges.contains_key(&(edge.1, edge.0)) {
            nb_in += 1;
        }
    }
    //
    let recall = nb_in as f64/ deleted_edges.len() as f64;
    let precision = nb_in as f64/ retrieved as f64;
    log::debug!("one_precision_iteration : precison {:3.e}, recall : {:3.e} nb_deleted : {}, nb_retrieved : {}", precision, recall, deleted_edges.len(), retrieved);
    //
    (precision,recall)
} // end of one_precision_iteration



/// estimate precision on nbiter iterations and delete_proba
/// Precision estimation is a costly operation as it requires computing the whole (nbnode, nbnode) matrix similarity
/// between embedded nodes and sorting them to compare with the deleted edges. So precision uses a comparison of each deleted edge
/// with most probables inexistant edges.  
/// 
/// AUC instead samples, for each deleted edge, a random inexistant edge from the original graph and compute its probability in the embedded graph
/// count number of times the deleted edge is more probable than the deleted.
/// 
pub fn estimate_precision<F : Copy + Sync, G, E : EmbeddedT<G> + std::marker::Sync>(csmat : &CsMatI<F, usize>, nbiter : usize, delete_proba : f64, 
                    symetric : bool, embedder : & dyn Fn(TriMatI<F, usize>) -> E) -> (Vec<f64>, Vec<f64>) {
    let rng = Xoshiro256PlusPlus::seed_from_u64(0);
    //
    let mut precision = Vec::<f64>::with_capacity(nbiter);
    let mut recall = Vec::<f64>::with_capacity(nbiter);
    // TODO can be made //
    for _ in 0..nbiter {
        let mut new_rng = rng.clone();
        new_rng.jump();
        let (iter_prec, iter_recall) = one_precision_iteration(csmat, delete_proba,  symetric, embedder, &mut new_rng);
        precision.push(iter_prec);
        recall.push(iter_recall);
    }
    //
    (precision,recall)
} // end of estimate_precision





/// estimate AUC as described in Link Prediction in complex Networks : A survey
///             Lü, Zhou. Physica 2011
fn one_auc_iteration<F, G, E>(csmat : &CsMatI<F, usize>, delete_proba : f64, symetric : bool, 
            embedder : &dyn Fn(TriMatI<F, usize>) -> E, rng : &mut Xoshiro256PlusPlus) -> f64
            where   F : Copy + std::marker::Sync,
                    E : EmbeddedT<G> + std::marker::Sync {
        //
    let mut auc : f64 = 0.;
    let nb_sample = 1000;
    let nbgood_order = 0;
    //
    // compute score of all non observed edges (ie.  nb_nodes*(nb_nodes-1)/2 - filtered out)
    // sample nb_sample 2-uples of edges one from the deleted edge and one inexistent (not in csmat)
    // count the number of times the distance of the first is less than the second.
    // filter
    let (mut trimat, deleted_edges) = filter_csmat(csmat, delete_proba, symetric, rng);
    // need to store trimat index before move to embedding
    let mut trimat_set = HashSet::<(usize,usize)>::with_capacity(trimat.nnz());
    for triplet in trimat.triplet_iter() {
        trimat_set.insert((triplet.1.0, triplet.1.1));
    }               
    //
    // embed (to be passed as a closure)
    //
    let embedded = &embedder(trimat);
    let mut good = 0.;
    //
    let nb_deleted = deleted_edges.len();
    let nb_nodes = csmat.shape().0;
    // as we can have large graph , mostly sparse to sample an inexistent edge we sample until we are outside csmat edges
    let del_uniform = Uniform::<usize>::from(0..nb_deleted);
    let node_random = Uniform::<usize>::from(0..nb_nodes);
    for k in 0..nb_sample {
        let del_edge = deleted_edges.get_index(del_uniform.sample(rng)).unwrap();
        let no_edge = loop {
            let i = node_random.sample(rng);
            let j = node_random.sample(rng);
            if i != j && !trimat_set.contains(&(i,j)) && deleted_edges.get_index_of(&(i,j)).is_none() {
                // edge (i,j) not on diagonal and neither in trimat set neither in deleted_edges, so inexistent edge
                break (i,j);
            }
        };
        let dist_del_edge = embedded.get_noderank_distance(del_edge.0,del_edge.1);
        let dist_no_edge = embedded.get_noderank_distance(no_edge.0,no_edge.1);
        if dist_del_edge < dist_no_edge {
            good += 1.;
        }
        else if dist_del_edge == dist_no_edge {
            good += 0.5;
        }
    }
    let auc = good / nb_sample as f64;
    log::debug!(" auc = {:3.e}", auc);
    //
    return auc;
} // end of one_auc_iteration



pub fn estimate_auc<F, G, E>(csmat : &CsMatI<F, usize>, nbiter : usize, delete_proba : f64, symetric : bool, 
            embedder : &dyn Fn(TriMatI<F, usize>) -> E) -> Vec<f64>
    where   F : Copy + std::marker::Sync,
            E : EmbeddedT<G> + std::marker::Sync {
        //
        let rng = Xoshiro256PlusPlus::seed_from_u64(0);
        //
        let mut auc = Vec::<f64>::with_capacity(nbiter);
        // TODO can be made //
        for _ in 0..nbiter {
            let mut new_rng = rng.clone();
            new_rng.jump();
            let iter_auc = one_auc_iteration(csmat, delete_proba,  symetric, embedder, &mut new_rng);
            auc.push(iter_auc);
        }
        //
        auc
} // end of estimate_auc



//================================================================================================================

mod tests {



//    cargo test csv  -- --nocapture
//    cargo test validation::link::tests::test_name -- --nocapture
//    RUST_LOG=graphembed::validation::link=TRACE cargo test link -- --nocapture

    use super::*;

    use indexmap::IndexSet;

    use crate::io::csv::*;
    use crate::sketching::nodesketch::*;
    use crate::embedding::{Embedded};

    use sprs::{TriMatI, CsMatI};

    use std::path::{Path};

    #[allow(unused_imports)]  // rust analyzer pb we need it!
    use ndarray::{array};
    
    #[allow(unused)]
    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }  

    // TODO should use the embedder trait
    // makes a (symetric) nodesketch Embedded to be sent to precision computations
    fn nodesketch_get_embedded(trimat : TriMatI<f64, usize>) -> Embedded<usize> {
        let sketch_size = 300;
        let decay = 0.001;
        let nb_iter = 4;
        let parallel = false;
        // now we embed
        let mut nodesketch = NodeSketch::new(sketch_size, decay, nb_iter, parallel, trimat);
        let embed_res = nodesketch.compute_embedded();
        embed_res.unwrap()
    } // end nodesketch_Embedded



    #[test]
    fn test_link_precision_nodesketch_lesmiserables() {
        //
        log_init_test();
        //
        log::debug!("in link.rs test_nodesketch_lesmiserables");
        let path = Path::new(crate::DATADIR).join("moreno_lesmis").join("out.moreno_lesmis_lesmis");
        log::info!("\n\n test_nodesketch_lesmiserables, loading file {:?}", path);
        let res = csv_to_trimat::<f64>(&path, false, b' ');
        if res.is_err() {
            log::error!("test_nodesketch_lesmiserables failed in csv_to_trimat");
            assert_eq!(1, 0);
        }
        else {
            let trimat_indexed = res.unwrap();
            let csrmat  : CsMatI<f64, usize> = trimat_indexed.0.to_csr();
            let symetric = true;
            let precision = estimate_precision(&csrmat, 10, 0.2, symetric, &nodesketch_get_embedded);
            log::info!("precision : {:?}", precision.0);
            log::info!("recall : {:?}", precision.1);
        };
    }  // end of test_link_precision_nodesketch_lesmiserables



    #[test]
    fn test_link_auc_nodesketch_lesmiserables() {
        //
        log_init_test();
        //
        log::debug!("in link.rs test_nodesketch_lesmiserables");
        let path = Path::new(crate::DATADIR).join("moreno_lesmis").join("out.moreno_lesmis_lesmis");
        log::info!("\n\n test_nodesketch_lesmiserables, loading file {:?}", path);
        let res = csv_to_trimat::<f64>(&path, false, b' ');
        if res.is_err() {
            log::error!("test_nodesketch_lesmiserables failed in csv_to_trimat");
            assert_eq!(1, 0);
        }
        else {
            let trimat_indexed = res.unwrap();
            let csrmat  : CsMatI<f64, usize> = trimat_indexed.0.to_csr();
            let symetric = true;
            let auc = estimate_auc(&csrmat, 10, 0.2, symetric, &nodesketch_get_embedded);
            log::info!("auc : {:?}", auc);
        };
    }  // end of test_link_auc_nodesketch_lesmiserables


}  // end of mod tests