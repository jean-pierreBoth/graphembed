//! do self loop augmentation

use sprs::{TriMatI, CsMatI};

// takes a matrix in a triplet form augment it by diagonal
pub(crate) fn diagonal_augmentation(graphtrip : &mut TriMatI<f64, usize>, weight : f64) -> CsMatI<f64, usize> {
    let shape = graphtrip.shape();
    log::debug!("diagonal_augmentation received shape {:?}", shape);
    assert_eq!(shape.0, shape.1);
    //
    for i in 0..shape.0 {
        graphtrip.add_triplet(i,i, weight);
    }
    //
    graphtrip.to_csr()
} // end of diagonal_augmentation