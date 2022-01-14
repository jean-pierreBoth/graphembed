//! this file implements the nodesketch algorithm 
//! described (nodesketch)<https://dl.acm.org/doi/10.1145/3292500.3330951>
//! 


#![allow(unused)]

use ndarray::{Array2, Array1};
use sprs::{TriMatI, CsMatI};

use ahash::{AHasher};
use std::collections::HashMap;
use probminhash::probminhasher::*;

use crate::embedding::Embedding;

use super::sla::*;

/// Compute the sketch of node proximity for a (undirected) graph.
/// sketch vector of a node is a list of integers obtained by hashing the weighted list of it neighbours (neigbour, weight)
/// The iterations consists in iteratively constructing new weighted list by combining initial adjacency list and successive weighted lists
pub struct NodeSketch {
    /// size of the skecth
    sketch_size: usize,
    /// Row compressed matrix representing self loop augmented graph i.e initial neighbourhood info
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
    /// TODO Loop on i can be made parallel
    fn sketch_slamatrix(&mut self) {
        // We use probminhash3a, allocate a Hash structure
        for i in 0..self.csrmat.rows() {
            let mut probminhash3 = ProbMinHash3::<usize, AHasher>::new(self.sketch_size, 0);
            for j in 0..self.csrmat.cols() {
                let w = self.csrmat.get_outer_inner(i,j).unwrap();
                probminhash3.hash_item(j,*w);
            }
            let sketch = Array1::from_vec(probminhash3.get_signature().clone());
            sketch.move_into(self.previous_sketches.row_mut(i));
        }
    } // end of sketch_slamatrix



    fn iterate(&mut self,  nb_iter:usize) {
        // first iteration, we fill previous sketches
        self.sketch_slamatrix();    
        for _ in 0..nb_iter {
            self.iteration();  
        }    
    }  // end of iterate


    // do iteration on sketches
    // TODO loop on rows must be made parallel, needs just a lock on self.sketches
    fn iteration(&mut self) {
        // now we repeatedly merge csrmat (loop augmented matrix) with sketches
        for (row, row_vec) in self.csrmat.outer_iterator().enumerate() {
            // new neighbourhood for current iteration 
            let mut v_k = HashMap::<usize, f64, ahash::RandomState>::default();
            let weight = self.decay/self.sketch_size as f64;
            // get an iterator on neighbours of node corresponding to row 
            let mut row_iter = row_vec.iter();
            while let Some(neighbour) = row_iter.next() {
                match v_k.get_mut(&neighbour.0) {
                    Some(val) => { *val = *val + weight * *neighbour.1; }
                    None              => { v_k.insert(neighbour.0, *neighbour.1).unwrap(); }
                };
                // get sketch of neighbour
                let neighbour_sketch = self.previous_sketches.row(neighbour.0);
                for n in neighbour_sketch {
                    match v_k.get_mut(&neighbour.0) {
                        // neighbour sketch contribute with weight neighbour.1 * decay / sketch_size to 
                        Some(val)   => { *val = *val + weight; }
                        None                => { v_k.insert(*n, weight).unwrap(); }
                    };                    
                }
            }
            // once we have a new list of (nodes, weight) we sketch it to fill the row of new sketches and to compact list of neighbours
            // as we possibly got more.
            let mut probminhash3a = ProbMinHash3a::<usize, AHasher>::new(self.sketch_size, 0);
            probminhash3a.hash_weigthed_hashmap(&v_k);
            let sketch = Array1::from_vec(probminhash3a.get_signature().clone());
            // save sketches into previous sketch
            sketch.move_into(self.sketches.row_mut(row));
        }  // end of for on row
        // TODO transfert to previous sketch.
    } // end of iteration



} // end of impl NodeSketch




//=====================================================================================================

mod tests {
 
use log::*;



#[allow(dead_code)]
fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}

use super::*; 




}  // end of mod tests