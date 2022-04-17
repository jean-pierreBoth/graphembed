//! do self loop augmentation

use sprs::{TriMatI, CsMatI};

use ndarray::{Array1};

// takes a matrix in a triplet form augment it by diagonal
pub(crate) fn diagonal_augmentation(graphtrip : &mut TriMatI<f64, usize>, weight : f64) -> CsMatI<f64, usize> {
    let shape = graphtrip.shape();
    log::debug!("diagonal_augmentation received shape {:?}", shape);
    assert_eq!(shape.0, shape.1);
    // determines max for each row
    let mut rowmax = Array1::<f64>::zeros(shape.0);
    let mut trip_iter = graphtrip.triplet_iter();
    while let Some(triplet) =  trip_iter.next() {
        if *triplet.0 > rowmax[triplet.1.0] {
            rowmax[triplet.1.0] = *triplet.0
        }
    }
    //
    for i in 0..shape.0 {
        // we do diagonal augmentation only for non isolated point
        if rowmax[i] > 0. && weight > 0. {
            // TODO could add rowmax ?
            graphtrip.add_triplet(i,i, weight);
//            log::trace!("diagonal_augmentation row : {}, value : {}", i, 1.);
        }
    }
    //
    graphtrip.to_csr()
} // end of diagonal_augmentation