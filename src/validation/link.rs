//! a simple link prediction to test embeddings


/// The scoring function for similarity is based upon the distance relative to the embedding being tested
/// Jaccard for nodesketch and L2 for Atp
/// We first implement precision measure as described in 
///       - Link Prediction in complex Networks : A survey
///             Lü, Zhou. Physica 2011


use log::*;
use anyhow::{anyhow};

use sprs::{TriMatI, CsMat};

// filter out edge with proba  delete_proba
fn filter_trimat(trimat : &TriMatI<F, usize>, delete_proba : f64) -> TrimatI<F,usize> {

} // end of filter_trimat


/// filters at rate delete_proba.
/// embed the filtered 
/// sort all (seen and not seen) edges according to embedding similarity function and return 
/// ratio of really filtered out / over the number of deleted edges (i.e precision of the iteration)
fn one_precision_iteration(trimat : &TriMatI<F, usize>, delete_proba : f64) -> f64 {

} // end of one_precision_iteration

/// estimate precision on nbiter iterations and delete_proba
fn estimate_precision(trimat : &TriMatI<F, usize>, nbiter : usize, delete_proba : f64) -> Vec<f64> {
} // end of estimate_precision