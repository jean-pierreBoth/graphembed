//! this file implements a small extension of the nodesketch algorithm for directed graph
//! described (nodesketch)<https://dl.acm.org/doi/10.1145/3292500.3330951>
//! 


#![allow(unused)]

use ndarray::{Array2, Array1};
use sprs::{TriMatI, CsMatI};

use ahash::{AHasher};
use std::collections::HashMap;
use probminhash::probminhasher::*;

use crate::embedding::EmbeddingAsym;

use super::sla::*;

/// Compute the sketch of node proximity for a directed graph.
/// sketch vector of a node is a list of integers obtained by hashing the weighted list of it neighbours (neigbour, weight)
/// The iterations consists in iteratively constructing new weighted list by combining initial adjacency list and successive weighted lists
pub struct NodeSketchAsym {
    /// size of the skecth
    sketch_size: usize,
    /// Row compressed matrix representing self loop augmented graph i.e initial neighbourhood info
    csrmat : CsMatI<f64, usize>,
    /// exponential decay coefficient for reducing weight of 
    decay : f64,
    /// The matrix storing all sketches along iterations for neighbours directed toward current node
    sketches_in : Array2<usize>,
    ///
    previous_sketches_in : Array2<usize>, 
    /// The matrix storing all sketches along iterations for neighbours reached from current node
    sketches_out : Array2<usize>, 
    /// sketches_out state at previous iterations 
    previous_sketches_out : Array2<usize>, 

} // end of struct NodeSketchAsym


impl NodeSketchAsym {

    pub fn new(sketch_size : usize, decay : f64, trimat : &mut  TriMatI<f64, usize>) -> Self {
        // TODO can adjust weight depending on context?
        let csrmat = diagonal_augmentation(trimat, 1.);
        let sketches_in = Array2::zeros((csrmat.cols(), sketch_size));
        let previous_sketches_in = Array2::zeros((csrmat.cols(), sketch_size));
        let sketches_out = Array2::zeros((csrmat.cols(), sketch_size));
        let previous_sketches_out = Array2::zeros((csrmat.cols(), sketch_size));
        NodeSketchAsym{sketch_size, decay, csrmat, sketches_in, previous_sketches_in, sketches_out, previous_sketches_out}
    }
    
    /// sketch of a row of initial self loop augmented matrix. Returns a vector of size self.sketch_size
    fn sketch_slamatrix(&mut self) {
        panic!("not yet implemented");
    } // end of sketch_slamatrix



    fn iterate(&mut self,  nb_iter:usize) {
        // first iteration, we fill previous sketches
        self.sketch_slamatrix();    
        for _ in 0..nb_iter {
            self.iteration();  
        }    
    }  // end of iterate


    // do iteration on sketches separately for in neighbours and out neighbours
    fn iteration(&mut self) {
        panic!("not yet implemented");
    } // end of iteration



} // end of impl NodeSketchAsym





//=====================================================================================================

mod tests {
 
use log::*;



#[allow(dead_code)]
fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}

use super::*; 




}  // end of mod tests