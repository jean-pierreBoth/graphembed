//! this file implements the nodesketch algorithm 
//! described [nodesketch)[https://dl.acm.org/doi/10.1145/3292500.3330951]
//! 




use ndarray::{Array2, Array1};
use sprs::{TriMatI, CsMatI};

use ahash::{AHasher};
use std::collections::HashMap;
use probminhash::probminhasher::*;


/// Compute the sketch of node proximity for a graph.
pub struct NodeSketch {
    /// size of the skecth
    sketch_size: usize,

    /// Row compressed matrix representing self loop augmented graph.
    csrmat : CsMatI<f64, usize>,
    /// exponential decay coefficient for reducing weight of 
    decay : f64,
    /// The matrix storing all sketches along iterations
    sketches : Array2<usize>,
    ///
    previous_sketches : Array2<usize>,  
} // end of struct NodeSketch


impl NodeSketch {

    pub fn new(sketch_size : usize, decay : f64, trimat : &mut  TriMatI<f64, usize>) -> Self {
        // TODO can adjust weight depending on context?
        let csrmat = diagonal_augmentation(trimat, 1.);
        let sketches = Array2::zeros((csrmat.cols(), sketch_size));
        let previous_sketches = Array2::zeros((csrmat.cols(), sketch_size));
        NodeSketch{sketch_size, decay, csrmat, sketches, previous_sketches}
    }
    
    /// sketch of a row of initial self loop augmented matrix. Returns a vector of size self.sketch_size
    fn sketch_rowsla(&mut self, row : usize) -> Array1<usize> {
        // We use probminhash3a, allocate a Hash structure
        let mut probminhash3 = ProbMinHash3::<usize, AHasher>::new(self.sketch_size, 0);
        for j in 0..self.csrmat.cols() {
            let w = self.csrmat.get_outer_inner(row,j).unwrap();
            probminhash3.hash_item(j,*w);
        }
        let sketch = Array1::from_vec(probminhash3.get_signature().clone());
        sketch
    } // end of sketch_row


    /// sketch a row of matrix of sketch. A row here has length sketch size with possible repetions of indexes
    /// so we need to use a Hashmap to get an association node -> weight
    /// We return a new vectors of usize length sketch_size
    fn sketch_rowsketch(&mut self, sketches : &Array2<usize> , row : usize, sketch_size : usize) {
        // We use probminhash3a, allocate a Hash structure
        let mut probminhash3 = ProbMinHash3::<usize, AHasher>::new(sketch_size, 0);
    } // end of sketch_rowsketch


    // do initialization of sketches
    fn iteration(&mut self) {
        // first iteration, we fill sketches
        for i in 0..self.csrmat.rows() {
            let rowsketch = self.sketch_rowsla(i);
            rowsketch.move_into(self.sketches.row_mut(i));
        }
        // now we repeatedly merge csrmat (loop augmented matrix) with sketches
        for (row, row_vec) in self.csrmat.outer_iterator().enumerate() {
            // get an iterator on neighbours of node corresponding to row 
            let mut row_iter = row_vec.iter();
            while let Some(neighbour) = row_iter.next() {
                // get sketch of neighbour
                let neighbour_sketch = self.sketches.row(neighbour.0);
                // neighbour sketch contribute with weight neighbour.1 * decay / sketch_size to 

            }
        }
        
    } // end of iteration



} // end of impl NodeSketch



// takes a matrix in a triplet form augment it by diagonal
fn diagonal_augmentation(graphtrip : &mut TriMatI<f64, usize>, weight : f64) -> CsMatI<f64, usize> {
    let shape = graphtrip.shape();
    assert_eq!(shape.0, shape.1);
    //
    for i in 0..shape.0 {
        graphtrip.add_triplet(i,i, weight);
    }
    //
    graphtrip.to_csr()
} // end of diagonal_augmentation


