//!  Generic jaccard function 
//! 


use ndarray::{ArrayView1};

/// The distance corresponding to nodesketch embedding
/// similarity is obtained by 1. - jaccard
// The hash signature is initialized in our use of Probminhash by a usize::MAX a rank of node clearly which cannot be encountered
pub(crate) fn jaccard_distance<T:Eq>(v1:&ArrayView1<T>, v2 : &ArrayView1<T>) -> f64 {
    assert_eq!(v1.len(), v2.len());
    let common = v1.iter().zip(v2.iter()).fold(0usize, |acc, v| if v.0 == v.1 { acc + 1 } else {acc});
    1.- (common as f64)/(v1.len() as f64)
} // end of jaccard