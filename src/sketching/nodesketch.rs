//! this file implements the nodesketch (symetric) algorithm
//!  
//! An asymetric implementation is provided in [nodesketchasym](super::nodesketchasym)
//! 


#[allow(unused)]
use anyhow::{anyhow};

use log::log_enabled;
use ndarray::{Array2, Array1, ArrayView1};
use sprs::{TriMatI, CsMatI};

use ahash::{AHasher};
use std::collections::HashMap;
use probminhash::probminhasher::*;
//
use rayon::iter::{ParallelIterator,IntoParallelIterator};
use parking_lot::{RwLock};
use std::sync::Arc;

use std::time::{SystemTime};
use cpu_time::ProcessTime;

//
use crate::embedding::{Embedded, EmbedderT};
use super::{sla::*, params::NodeSketchParams};


/// The distance corresponding to nodesketch embedding
/// similarity is obtained by 1. - jaccard
// The hash signature is initialized in our use of Probminhash by a usize::MAX a rank of node clearly which cannot be encountered
pub(crate) fn jaccard_distance(v1:&ArrayView1<usize>, v2 : &ArrayView1<usize>) -> f64 {
    assert_eq!(v1.len(), v2.len());
    let common = v1.iter().zip(v2.iter()).fold(0usize, |acc, v| if v.0 == v.1 { acc + 1 } else {acc});
    1.- (common as f64)/(v1.len() as f64)
} // end of jaccard

// We access rows in //, we need rows to be protected by a RwLock
// as Rows are accessed once in each iteration we avoid deadlocks
// But we need RwLock to get Sync for struct NodeSketch in closures
pub type RowSketch = Arc<RwLock<Array1<usize>>>;





/// Compute the sketch of node proximity for a (undirected) graph.  
/// sketch vector of a node is a list of integers obtained by hashing the weighted list of it neighbours (neigbour, weight).
/// The iterations consists in iteratively constructing new weighted list by combining initial adjacency list and successive hashed weighted lists
pub struct NodeSketch {
    /// specific arguments
    params : NodeSketchParams,
    /// Row compressed matrix representing self loop augmented graph i.e initial neighbourhood info
    csrmat : CsMatI<f64, usize>,
    /// The vector storing node sketch along iterations, length is nbnodes, each RowSketch is a vecotr of sketch_size
    sketches : Vec<RowSketch>,
    ///
    previous_sketches : Vec<RowSketch>, 
} // end of struct NodeSketch


impl  NodeSketch {

    // We pass a Trimat as we have to do self loop augmentation
    pub fn new(params : NodeSketchParams, mut trimat :TriMatI<f64, usize>) -> Self {
        // TODO can adjust weight depending on context?
        let csrmat = diagonal_augmentation(&mut trimat, 1.);
        log::debug!(" NodeSketch new csrmat dims nb_rows {}, nb_cols {} ", csrmat.rows(), csrmat.cols());
        let mut sketches = Vec::<RowSketch>::with_capacity(csrmat.rows());
        let mut previous_sketches = Vec::<RowSketch>::with_capacity(csrmat.rows());
        for _ in 0..csrmat.rows() {
            let sketch = Array1::<usize>::zeros(params.get_sketch_size());
            sketches.push(Arc::new(RwLock::new(sketch)));
            let previous_sketch = Array1::<usize>::zeros(params.get_sketch_size());
            previous_sketches.push(Arc::new(RwLock::new(previous_sketch)));
        }
        NodeSketch{params , csrmat, sketches, previous_sketches}
    } // end of NodeSketch::new 
    

    /// get sketch_size 
    pub fn get_sketch_size(&self) -> usize {
        self.params.sketch_size
    } // end of get_nb_nodes


    /// get decay weight for multi hop neighbours
    pub fn get_decay_weight(&self) -> f64 {
        self.params.decay
    } // end of get_decay_weight


    /// get number of nodes
    pub fn get_nb_nodes(&self) -> usize {
        self.csrmat.rows()
    }  // end of get_nb_nodes


    // utility to debug very small tests
    #[allow(unused)]
    pub(crate) fn dump_row_iteration(&self, noderank : usize) {
        println!("row iteration i : {}", noderank);
        println!("previous state  {:?}",  self.previous_sketches[noderank].read());
        println!("new state  {:?} ", self.sketches[noderank].read());
    } // end of dump_state


    /// sketch of a row of initial self loop augmented matrix. Returns a vector of size self.sketch_size
    /// We initialize signatures with row so an isolated node will have just its identity as signature
    fn sketch_slamatrix(&mut self, parallel : bool) {
        // 
        let treat_row = | row : usize | {
            let mut probminhash3 = ProbMinHash3::<usize, AHasher>::new(self.get_sketch_size(), row);
            let col_range = self.csrmat.indptr().outer_inds_sz(row);
            log::trace!("sketch_slamatrix i : {}, col_range : {:?}", row, col_range);            
            for k in col_range {
                let j = self.csrmat.indices()[k];
                let w = self.csrmat.data()[k];
        //        log::trace!("sketch_slamatrix row : {}, k  : {}, col {}, w {}", row, k, j ,w);
                probminhash3.hash_item(j,w);
            }
            let sketch = probminhash3.get_signature();
            log::trace!(" sketch_slamatrix sketch row i : {} , sketch : {:?}", row, sketch);
            for j in 0..self.get_sketch_size() {
                self.previous_sketches[row].write()[j] = sketch[j];
            }
        };

        if !parallel {
            log::debug!(" not parallel case nb rows  {}",self.csrmat.rows()) ;
            for row in 0..self.csrmat.rows() {
                if self.csrmat.indptr().nnz_in_outer_sz(row) > 0 {
                    log::trace!("sketching row {}", row);
                    treat_row(row);
                }
            }
        } 
        else {
            // parallel case
            (0..self.csrmat.rows()).into_par_iter().for_each(| row|  if self.csrmat.indptr().nnz_in_outer_sz(row) > 0 { treat_row(row); })
        }
        log::debug!("sketch_slamatrix done")
    } // end of sketch_slamatrix



    /// computes the embedding 
    ///     - nb_iter  : corresponds to the number of hops we want to explore around each node.
    ///     - parallel : a flag to ask for parallel exploration of nodes neighbourhood 
    pub fn compute_embedded(&mut self) -> Result<Embedded<usize>,anyhow::Error> {
        log::debug!("in nodesketch::compute_Embedded");
        let cpu_start = ProcessTime::now();
        let sys_start = SystemTime::now();
        //
        let parallel = self.params.parallel;
        // first iteration, we fill previous sketches
        self.sketch_slamatrix(parallel);    
        for _ in 0..self.params.nb_iter {
            if parallel {
                self.parallel_iteration();
            }
            else {
                self.iteration();
            }
        }
        //
        println!(" embedding sys time(s) {:.2e} cpu time(s) {:.2e}", sys_start.elapsed().unwrap().as_secs(), cpu_start.elapsed().as_secs());
        // allocate the (symetric) Embedded
        let nbnodes = self.sketches.len();
        let dim = self.sketches[0].read().len();
        let mut embedded = Array2::<usize>::zeros((nbnodes,dim));
        for i in 0..nbnodes {
            for j in 0..self.get_sketch_size() {
                embedded.row_mut(i)[j] = self.sketches[i].read()[j];
            }
        }
        let embedded = Embedded::<usize>::new(embedded, jaccard_distance);
        //
        Ok(embedded)
    }  // end of compute_embedded


    // do serial iteration on sketches
    fn iteration(&mut self) {
        // now we repeatedly merge csrmat (loop augmented matrix) with sketches
        for (row, _) in self.csrmat.outer_iterator().enumerate() {
            // new neighbourhood for current iteration 
            self.treat_row(&row);
            if log_enabled!(log::Level::Trace) {
                log::trace!("dump end of iteration : ");
                self.dump_row_iteration(row);
            }
        }  // end of for on row
        // transfer sketches into previous sketches
        for i in 0..self.get_nb_nodes() { 
            let mut row_write = self.previous_sketches[i].write();
            for j in 0..self.get_sketch_size() {
                row_write[j] = self.sketches[i].read()[j];
            }
        }       
    } // end of iteration



    // do one iteration with // treatment of nodes
    fn parallel_iteration(&mut self) {
        // now we repeatedly merge csrmat (loop augmented matrix) with sketches
        (0..self.csrmat.rows()).into_par_iter().for_each( |row| self.treat_row(&row));
        // transfer sketches into previous sketches
        for i in 0..self.get_nb_nodes() { 
            let mut row_write = self.previous_sketches[i].write();
            for j in 0..self.get_sketch_size() {
                row_write[j] = self.sketches[i].read()[j];
            }
        }  
    } // end of parallel_iteration


    // given a row (its number and the data in compressed row Matrice corresponding) 
    // the function computes sketch value given previous sketch values
    fn treat_row(&self, row : &usize) {
        // if we have no data ar row we return immediately
        let row_vec = self.csrmat.outer_view(*row);
        if row_vec.is_none() {
            return;
        }
        let row_vec = row_vec.unwrap();
        // new neighbourhood for row at current iteration. Store neighbours of row with a weight. 
        let mut v_k = HashMap::<usize, f64, ahash::RandomState>::default();
        let weight = self.get_decay_weight()/self.get_sketch_size() as f64;
//        let weight = self.get_decay_weight();
        // get an iterator on neighbours of node corresponding to row 
        let mut row_iter = row_vec.iter();
        while let Some(neighbour) = row_iter.next() {
            // neighbour.0 is a neighbour of row, it is brought with the weight connection from row to neighbour
            match v_k.get_mut(&neighbour.0) {
                Some(val) => {
                    *val = *val + *neighbour.1;
                    log::trace!("{} augmenting weight in v_k for neighbour {},  new weight {:.3e}", 
                            neighbour.0, *neighbour.1, *val);  
                }
                None    => { 
                    log::trace!("adding node in v_k {}  weight {:.3e}", neighbour.0, *neighbour.1);
                    v_k.insert(neighbour.0, *neighbour.1); 
                }
            };
            // get component due to previous sketch of neighbour
            let neighbour_sketch = &*self.previous_sketches[neighbour.0].read();
            for n in neighbour_sketch {
            // something (here n) in a neighbour sketch is brought with the weight connection from row to neighbour multiplied by the decay factor
                match v_k.get_mut(n) {
                    // neighbour sketch contribute with weight neighbour.1 * decay / sketch_size to 
                    Some(val)   => { 
                            *val = *val + weight * *neighbour.1;
                            log::trace!("{} sketch augmenting node {} weight in v_k with decayed edge weight {:.3e} new weight {:.3e}", 
                                        neighbour.0 , *n, weight * *neighbour.1, *val);
                    }
                    None                =>  {
                        log::trace!("{} sketch adding node with {} decayed weight {:.3e}", neighbour.0 , *n, weight * *neighbour.1);
                        v_k.insert(*n, weight *neighbour.1 );
                    }
                };                    
            }
        }  // end of while
        // once we have a new list of (nodes, weight) we sketch it to fill the row of new sketches and to compact list of neighbours
        // We initialize with row itself, so that if we have an isolated node, signature is just itself. So all isolated nodes will be at
        // maximal distance of one another!
        let mut probminhash3a = ProbMinHash3a::<usize, AHasher>::new(self.get_sketch_size(), *row);
        probminhash3a.hash_weigthed_hashmap(&v_k);
        let sketch = Array1::from_vec(probminhash3a.get_signature().clone());
        // save sketches into self sketch
        let mut row_write = self.sketches[*row].write();
        for j in 0..self.get_sketch_size() {
            row_write[j] = sketch[j];
        }            
    }  // end of treat_row


} // end of impl NodeSketch



impl EmbedderT<usize> for NodeSketch {
    type Output = Embedded<usize>;
    ///
    fn embed(&mut self) -> Result<Embedded<usize>, anyhow::Error > {
        let res = self.compute_embedded();
        match res {
            Ok(embeded) => {
                return Ok(embeded);
            },
            Err(err) => { return Err(err);}
        }
    } // end of embed
} // end of impl<f64> EmbedderT<f64>



//=====================================================================================================

#[cfg(test)]
mod tests {



use super::*; 

use crate::prelude::*;

use crate::io::csv::csv_to_trimat;


fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}


// This small test can be run with RUST_LOG=graphite=trace cargo test test_nodesketch_lesmiserables -- --nocapture >log 2>&1
// to trace sketch evolutions with iterations and see the impact of decay weight.
#[test]
fn test_nodesketch_lesmiserables() {
    log_init_test();
    //
    log::debug!("in nodesketch.rs test_nodesketch_lesmiserables");
    let path = std::path::Path::new(crate::DATADIR).join("moreno_lesmis").join("out.moreno_lesmis_lesmis");
    log::info!("\n\n test_nodesketch_lesmiserables, loading file {:?}", path);
    let res = csv_to_trimat::<f64>(&path, false, b' ');
    if res.is_err() {
        log::error!("test_nodesketch_lesmiserables failed in csv_to_trimat");
        assert_eq!(1, 0);
    }
    let (trimat, node_index) = res.unwrap();
    let sketch_size = 15;
    let decay = 100.;
    let nb_iter = 2;
    let parallel = false;
    let symetric = true;
    let params = NodeSketchParams{sketch_size, decay, nb_iter, symetric, parallel};
    // now we embed
    let mut nodesketch = NodeSketch::new(params, trimat);
    let sketch_embedding = Embedding::new(node_index, &mut nodesketch);
    if sketch_embedding.is_err() {
        log::error!("test_nodesketch_lesmiserables failed in compute_Embedded");
        assert_eq!(1, 0);        
    }
    let embed_res = sketch_embedding.unwrap();
    // compute some distances
    // get distance between node 11 and 27 (largest edge proximity weight in file = 31)
    let dist_11_27  = embed_res.get_node_distance(11, 27);
    log::debug!("node (11,27)  =  rank({},{})" , embed_res.get_node_rank(11).unwrap(), embed_res.get_node_rank(27).unwrap());
    log::debug!("distance between nodes 11 and 27 : {}, weight in file {} ", dist_11_27, 31);
    let dist_11_33  = embed_res.get_node_distance(11, 33);
    log::debug!("distance between nodes 11 and 33 {} , weight in file : {}", dist_11_33, 1);

    // dump some vectors
    let embedded = embed_res.get_embedded_data();

    let rank = embed_res.get_node_rank(11).unwrap();
    log::debug!("\n\n row {:?} sketch {:?} ", rank, embedded.get_embedded().row(rank));

    let rank = embed_res.get_node_rank(27).unwrap();
    log::trace!("\n\n row {:?} sketch {:?} ", rank, embedded.get_embedded().row(rank));
    // nodeid  21 and 23 have the same neighbourhoods with very similar weights.
    let dist_21_23 =  embed_res.get_node_distance(21,23);
    log::debug!("embedded distance between nodes 21 and 23 : {:3.e}", dist_21_23);
    let rank = 48;
    log::trace!("\n\n row {:?} , node_id {:?}, sketch {:?} ", rank, embed_res.get_node_id(rank) ,embedded.get_embedded().row(rank));
    let rank = 50;
    log::trace!("\n\n row {:?},  node_id {:?}, sketch {:?} ", rank, embed_res.get_node_id(rank) ,embedded.get_embedded().row(rank));
    // check for rank 26 27 nodes (35,36) same neighborhood, same weights , must have dist <= 0.05
    let node_of_rank_26 = *embed_res.get_node_id(26).unwrap();
    let node_of_rank_27 = *embed_res.get_node_id(27).unwrap();
    let dist = embed_res.get_node_distance(node_of_rank_26,node_of_rank_27);
    log::debug!("distance between nodes n1 : {} n2 : {}, dist : {:3.e}", node_of_rank_26, node_of_rank_27, dist);
} // enf of test_nodesketch_lesmiserables



}  // end of mod tests