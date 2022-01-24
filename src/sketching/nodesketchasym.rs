//! this file implements a small extension of the nodesketch algorithm for directed graph
//! The original algorith is implemented in ['NodeSketch`]  and described in 
//! (nodesketch)<https://dl.acm.org/doi/10.1145/3292500.3330951>
//! 


#![allow(unused)]

use anyhow::{anyhow};

use ndarray::{Array2, Array1};
use sprs::{TriMatI, CsMatI, CsVecBase};

use ahash::{AHasher};
use std::collections::HashMap;
use probminhash::probminhasher::*;

use rayon::iter::{ParallelIterator,IntoParallelRefIterator};
use parking_lot::{RwLock};
use std::sync::Arc;


use crate::embedding::EmbeddingAsym;

use super::sla::*;


type RowSketch = Arc<RwLock<Array1<usize>>>;

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
    sketches_in : Vec<RowSketch>,
    ///
    previous_sketches_in : Vec<RowSketch>, 
    /// The matrix storing all sketches along iterations for neighbours reached from current node
    sketches_out : Vec<RowSketch>, 
    /// sketches_out state at previous iterations 
    previous_sketches_out : Vec<RowSketch>, 

} // end of struct NodeSketchAsym


impl NodeSketchAsym {

    pub fn new(sketch_size : usize, decay : f64, trimat : &mut  TriMatI<f64, usize>) -> Self {
        // TODO can adjust weight depending on context?
        let csrmat = diagonal_augmentation(trimat, 1.);
        let mut sketches_in = Vec::<RowSketch>::with_capacity(csrmat.rows());
        let mut previous_sketches_in = Vec::<RowSketch>::with_capacity(csrmat.rows());
        let mut sketches_out = Vec::<RowSketch>::with_capacity(csrmat.rows());
        let mut previous_sketches_out = Vec::<RowSketch>::with_capacity(csrmat.rows());
        for _ in 0..csrmat.rows() {
            let sketch = Array1::<usize>::zeros(sketch_size);
            sketches_in.push(Arc::new(RwLock::new(sketch)));
            let previous_sketch_in = Array1::<usize>::zeros(sketch_size);
            previous_sketches_in.push(Arc::new(RwLock::new(previous_sketch_in)));
        }
        NodeSketchAsym{sketch_size, decay, csrmat, sketches_in, previous_sketches_in, sketches_out, previous_sketches_out}
    }  // end of for NodeSketchAsym::new
    

    /// get sketch_size 
    pub fn get_sketch_size(&self) -> usize {
        self.sketch_size
    } // end of get_nb_nodes


    /// get decay weight for multi hop neighbours
    pub fn get_decay_weight(&self) -> f64 {
        self.decay
    } // end of get_decay_weight


    /// get number of nodes
    pub fn get_nb_nodes(&self) -> usize {
        self.csrmat.rows()
    }  // end of get_nb_nodes
    
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


    // do iteration on sketches separately for in neighbours and out neighbours
    fn parallel_iteration(&mut self) {
        panic!("not yet implemented");
    } // end of parallel_iteration


    // given a row (its number and the data in compressed row Matrice corresponding) 
    // the function omputes sketch value given previous sketch values
    fn treat_row(&self, row : &usize, row_vec : &CsVecBase<&[usize], &[f64], f64, usize>) {
        panic!("not yet implemented");
    } // end of treat_row


    /// computes the embedding. 
    ///  - nb_iter  : corresponds to number of hops explored around a node.  
    ///  - parallel : if true each iteration treats node in parallel
    pub fn compute_embedding(&mut self, nb_iter:usize, parallel : bool) -> Result<EmbeddingAsym<usize>, anyhow::Error> {
        // first iteration, we fill previous sketches
        self.sketch_slamatrix();    
        for _ in 0..nb_iter {
            if parallel {
                self.parallel_iteration();
            }
            else {
                self.iteration(); 
            }
        }
        // allocate the asymetric embedding
        //
        return Err(anyhow!("not yet implemented"));
    } // end of compute_embedding

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