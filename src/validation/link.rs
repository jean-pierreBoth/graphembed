//! a simple link prediction to test embeddings


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

use sprs::{TriMatI, CsMatI};

// filter out edge with proba  delete_proba
fn filter_csmat<F>(csrmat : &CsMatI<F, usize>, delete_proba : f64, rng : &mut Xoshiro256PlusPlus)  {
    //
    assert!(delete_proba < 1. && delete_proba > 0.);
    //
    let uniform = Uniform::<f64>::new(0., 1.);

    let nb_nodes = csrmat.rows();

    let mut rows = Vec::<usize>::with_capacity(nb_nodes);
    let mut cols = Vec::<usize>::with_capacity(nb_nodes);
    let mut values = Vec::<F>::with_capacity(nb_nodes);


    let mut csmat_iter = csrmat.iter();
    while let Some(t) = csmat_iter.next()  {
        let xsi = uniform.sample(rng);
        if xsi > delete_proba {
            // we keep the triplet
        }
        else {
            // we must check we do not make an isolated node
        }

    }


} // end of filter_trimat


/// filters at rate delete_proba.
/// embed the filtered 
/// sort all (seen and not seen) edges according to embedding similarity function and return 
/// ratio of really filtered out / over the number of deleted edges (i.e precision of the iteration)
fn one_precision_iteration<F>(csmat : &CsMatI<F, usize>, delete_proba : f64, rng : Xoshiro256PlusPlus) -> f64 {

    return -1.;

} // end of one_precision_iteration



/// estimate precision on nbiter iterations and delete_proba
/// Precision estimation is a costly operation as it requires computing the whole (nbnode, nbnode) matrix similarity
/// between embedded nodes and sorting them to compare with the deleted edges.
/// AUC use instead sampling a random inexistant edge from the original graph and compute its probabiity in the embedded graph.
/// 
fn estimate_precision<F>(csmat : &CsMatI<F, usize>, nbiter : usize, delete_proba : f64) -> Vec<f64> {
    let rng = Xoshiro256PlusPlus::seed_from_u64(0);
    //
    let mut precision = Vec::<f64>::with_capacity(nbiter);
    // TODO can be made //
    for _ in 0..nbiter {
        let mut new_rng = rng.clone();
        new_rng.jump();
        let prec = one_precision_iteration(csmat, delete_proba, new_rng);
        precision.push(prec);
    }
    //
    precision
} // end of estimate_precision