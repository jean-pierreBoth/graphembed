//! a simple link prediction to test embeddings

#![allow(unused)]


/// The scoring function for similarity is based upon the distance relative to the embedding being tested
/// Jaccard for nodesketch and L2 for Atp
/// We first implement precision measure as described in 
///       - Link Prediction in complex Networks : A survey
///             Lü, Zhou. Physica 2011


use log::*;
use anyhow::{anyhow};

use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;
use rand::distributions::{Uniform, Distribution};

use std::collections::HashSet;

use sprs::{TriMatI, CsMatI};
use ndarray::{Array1, Array2};

use rayon::prelude::*;
use parking_lot::RwLock;

use crate::embedding::{Embedding, EmbeddingT};
use crate::embedder::EmbedderT;
use crate::sketching::{nodesketch::*, nodesketchasym};
use crate::atp::*;

// filter out edge with proba delete_proba
// The return a tuple containing the matrix of the graph with some edge deleted and a perfect hash on deleted edges
fn filter_csmat<F>(csrmat : &CsMatI<F, usize>, delete_proba : f64, rng : &mut Xoshiro256PlusPlus) -> (TriMatI::<F, usize>, HashSet<(usize,usize)>)
    where F: Copy {
    //
    assert!(delete_proba < 1. && delete_proba > 0.);
    // get total in + out degree
    let mut degree = csrmat.degrees();
    let nb_nodes = csrmat.rows();
    let nb_edge = csrmat.nnz();
    let mut deleted_pairs = Vec::<(usize, usize)>::with_capacity((nb_edge as f64 * delete_proba).round() as usize);
    let mut deleted_edge = HashSet::with_capacity((nb_edge as f64 * delete_proba).round() as usize);
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
    //
    while let Some((value,(row, col))) = csmat_iter.next()  {
        let xsi = uniform.sample(rng);
        discard = false;
        if xsi < delete_proba {
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
        }
    }  // end while
    //
    log::info!(" ratio discarded = {:}", discarded as f64/ nb_nodes as f64);
    log::info!(" not discarded due to low degree {:?}", not_discarded);
    //
    let trimat = TriMatI::<F, usize>::from_triplets((nb_nodes,nb_nodes), rows, cols, values);
    //
    (trimat, deleted_edge)
} // end of filter_csmat

// local edge type corresponding to node1, ndde2 , distance from node1 to node2
struct Edge(usize,usize,f64);


/// filters at rate delete_proba.
/// embed the filtered 
/// sort all (seen and not seen) edges according to embedding similarity function and return 
/// ratio of really filtered out / over the number of deleted edges (i.e precision of the iteration)
fn one_precision_iteration<F: Copy, E : EmbeddingT<F> + std::marker::Sync>(csmat : &CsMatI<F, usize>, delete_proba : f64, embedder : &dyn Fn(&TriMatI<F, usize>) -> E, rng : &mut Xoshiro256PlusPlus) -> f64 {
    // filter
    let (mut trimat, deleted_edges) = filter_csmat(csmat, delete_proba,rng);
    // embed (to be passed as a closure)
    let embedding = &embedder(&trimat);
    // compute all similarities betwen nodes pairs and sort
    let nb_nodes = embedding.get_nb_nodes();
    let mut embedded_edges = Vec::<Edge>::with_capacity(nb_nodes*nb_nodes);
    let f_i = |i : usize| -> Vec<Edge> {
        (0..nb_nodes).into_iter().map(|j| Edge{0:i, 1:j, 2:embedding.get_node_distance(i,j)}).collect()
    };
    let mut row_embedded : Vec<Vec<Edge>> = (0..nb_nodes).into_par_iter().map(|i| f_i(i)).collect();
    for i in 0..nb_nodes {
        embedded_edges.append(&mut row_embedded[i]);
    }
    // sort embedded_edges in distance increasing order and keep the as many as the number we deleted. (keep the most probable)
    embedded_edges.sort_unstable_by(|ea, eb| ea.2.partial_cmp(&eb.2).unwrap());
    embedded_edges.truncate(deleted_edges.len());
    // find how many deleted edges are in upper part of the sorted edges.
    let nb_in = embedded_edges.iter().fold(0usize, |acc, edge| if deleted_edges.contains(&(edge.0, edge.1)) { acc+1} else {acc});
    // for edge in embedded_edges {
    //     let is_in = deleted_edges.contains(&(edge.0, edge.1));
    // }
    //
    let precision = nb_in as f64/ deleted_edges.len() as f64;
    log::debug!("one_precision_iteration : precison {}, nb_deleted : {}", precision, deleted_edges.len());
    //
    precision
} // end of one_precision_iteration



/// estimate precision on nbiter iterations and delete_proba
/// Precision estimation is a costly operation as it requires computing the whole (nbnode, nbnode) matrix similarity
/// between embedded nodes and sorting them to compare with the deleted edges. So precision uses a comparison of each deleted edge
/// with most probables inexistant edges.  
/// 
/// AUC instead samples, for each deleted edge, a random inexistant edge from the original graph and compute its probability in the embedded graph
/// count number of times the deleted edge is more probable than the deleted.
/// 
fn estimate_precision<F : Copy, E : EmbeddingT<F> + std::marker::Sync>(csmat : &CsMatI<F, usize>, nbiter : usize, delete_proba : f64, 
                    embedder : &dyn Fn(&TriMatI<F, usize>) -> E) -> Vec<f64> {
    let rng = Xoshiro256PlusPlus::seed_from_u64(0);
    //
    let mut precision = Vec::<f64>::with_capacity(nbiter);
    // TODO can be made //
    for _ in 0..nbiter {
        let mut new_rng = rng.clone();
        new_rng.jump();
        let prec = one_precision_iteration(csmat, delete_proba, embedder, &mut new_rng);
        precision.push(prec);
    }
    //
    precision
} // end of estimate_precision



mod tests {

    use super::*;
    
    #[allow(unused_imports)]  // rust analyzer pb we need it!
    use ndarray::{array};
    
    #[allow(unused)]
    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }  

}  // end of mod tests