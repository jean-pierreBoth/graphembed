//! this file implements the nodesketch (symetric) algorithm 
//! described (nodesketch)<https://dl.acm.org/doi/10.1145/3292500.3330951>
//! An asymetric implementation is in ['NodeSketchAsym`]
//! 



use ndarray::{Array2, Array1};
use sprs::{TriMatI, CsMatI, CsVecBase};

use ahash::{AHasher};
use std::collections::HashMap;
use probminhash::probminhasher::*;
//
use rayon::iter::{ParallelIterator,IntoParallelRefIterator};
use parking_lot::{RwLock};
use std::sync::Arc;

//
use crate::embedding::Embedding;
use super::sla::*;

// We access rows in //, we need rows to be protected by a RwLock
// as Rows are accesses once in each iteration we avoid locks
type RowSketch = Arc<RwLock<Array1<usize>>>;

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
    /// The vector storing node sketch along iterations, length is nbnodes, each RowSketch is a vecotr of sketch_size
    sketches : Vec<RowSketch>,
    ///
    previous_sketches : Vec<RowSketch>,  
} // end of struct NodeSketch


impl NodeSketch {

    pub fn new(sketch_size : usize, decay : f64, trimat : &mut  TriMatI<f64, usize>) -> Self {
        // TODO can adjust weight depending on context?
        let csrmat = diagonal_augmentation(trimat, 1.);
        let mut sketches = Vec::<RowSketch>::with_capacity(csrmat.rows());
        let mut previous_sketches = Vec::<RowSketch>::with_capacity(csrmat.rows());
        for _ in 0..csrmat.rows() {
            let sketch = Array1::<usize>::zeros(sketch_size);
            sketches.push(Arc::new(RwLock::new(sketch)));
            let previous_sketch = Array1::<usize>::zeros(sketch_size);
            previous_sketches.push(Arc::new(RwLock::new(previous_sketch)));
        }
        NodeSketch{sketch_size, decay, csrmat, sketches, previous_sketches}
    } // end of NodeSketch::new 
    

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


    // utility to debug very small tests
    #[allow(unused)]
    pub(crate) fn dump_state(&self) {
        println!("\n dump previous state \n");
        for i in 0..self.get_nb_nodes() {
            println!("row i : {}   {:?}", i, self.previous_sketches[i].read());
        }
        println!("\n dump state \n");
        for i in 0..self.get_nb_nodes() {
            println!("row i : {}   {:?}", i, self.sketches[i].read());
        }
    } // end of dump_state


    /// sketch of a row of initial self loop augmented matrix. Returns a vector of size self.sketch_size
    /// TODO Loop on i can also be made parallel
    fn sketch_slamatrix(&mut self) {
        // We use probminhash3a, allocate a Hash structure
        for i in 0..self.csrmat.rows() {
            let mut probminhash3 = ProbMinHash3::<usize, AHasher>::new(self.sketch_size, 0);
            for j in 0..self.csrmat.cols() {
                let w = self.csrmat.get_outer_inner(i,j).unwrap();
                probminhash3.hash_item(j,*w);
            }
            let sketch = probminhash3.get_signature();
            for j in 0..self.get_sketch_size() {
                self.previous_sketches[i].write()[j] = sketch[j];
            }
        }
    } // end of sketch_slamatrix



    pub fn compute_embedding(&mut self,  nb_iter:usize, parallel : bool) -> Result<Embedding<usize>,anyhow::Error> {
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
        // allocate the (symetric) embedding
        let nbnodes = self.sketches.len();
        let dim = self.sketches[0].read().len();
        let mut embedded = Array2::<usize>::zeros((nbnodes,dim));
        for i in 0..nbnodes {
            for j in 0..self.get_sketch_size() {
                embedded.row_mut(i)[j] = self.sketches[i].read()[j];
            }
        }
        let embedding = Embedding::<usize>::new(embedded);
        //
        Ok(embedding)
    }  // end of compute_embedding


    // do serial iteration on sketches
    fn iteration(&mut self) {
        // now we repeatedly merge csrmat (loop augmented matrix) with sketches
        for (row, row_vec) in self.csrmat.outer_iterator().enumerate() {
            // new neighbourhood for current iteration 
            self.treat_row(&row, &row_vec);
        }  // end of for on row
        // transfer sketches into previous sketches
        for i in 0..self.get_nb_nodes() { 
            let mut row_write = self.previous_sketches[i].write();
            for j in 0..self.get_sketch_size() {
                row_write[j] = self.sketches[i].read()[j];
            }
        }       
    } // end of iteration


    // given a row (its number and the data in compressed row Matrice corresponding) 
    // the function omputes sketch value given previous sketch values
    fn treat_row(&self, row : &usize, row_vec : &CsVecBase<&[usize], &[f64], f64, usize>) {
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
            let neighbour_sketch = &*self.previous_sketches[neighbour.0].read();
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
        // save sketches into self sketch
        let mut row_write = self.sketches[*row].write();
        for j in 0..self.get_sketch_size() {
            row_write[j] = sketch[j];
        }            
    }  // end of treat_row



    // do one iteration with // treatment of nodes
    fn parallel_iteration(&mut self) {
        // we must first gather all information on rows
        let nb_rows = self.csrmat.rows();
        let mut rows_info = Vec::<(usize,CsVecBase<&[usize], &[f64], f64, usize>) >::with_capacity(nb_rows);
        for (row, row_vec) in self.csrmat.outer_iterator().enumerate() {
            rows_info.push((row, row_vec));
        }
        // now we repeatedly merge csrmat (loop augmented matrix) with sketches
        rows_info.par_iter().for_each( |(row, row_vec)| self.treat_row(row, row_vec));
    } // end of parallel_iteration


} // end of impl NodeSketch




//=====================================================================================================

mod tests {


#[allow(unused)]
use log::*;

#[allow(unused)]
use super::*; 


#[allow(dead_code)]
fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}





}  // end of mod tests