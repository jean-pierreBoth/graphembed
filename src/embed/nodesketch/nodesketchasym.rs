//! this file implements an extension of the nodesketch algorithm for directed graph.  
//! The original algorithm is implemented in [nodesketch](super::nodesketchsym)
//!
//! The iterations consists in iteratively constructing new weighted list of neighbours of each node considered
//! once as a source and once as a target.  
//! An embedded node is represented by 2 vectors, one for the node considered as a source, one as a target.
//!
//! The final dissimilarity between 2 nodes is a mix of the distances between nodes considered as sources, as target and as linked
//! by a weighted edge if any.

// For a directed graph we suppose the Csr matrix so that mat[[i,j]] is edge from i to j.

#[allow(unused)]
use anyhow::anyhow;

use ndarray::{Array1, Array2};
use sprs::{CsMatI, TriMatI};

use ahash::AHasher;
use probminhash::probminhasher::*;
use std::collections::HashMap;

use parking_lot::RwLock;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;

use cpu_time::ProcessTime;
use std::time::SystemTime;

use super::{params::NodeSketchParams, sla::*};
use crate::embed::tools::degrees::*;
use crate::embedding::{EmbeddedAsym, EmbedderT};

pub type RowSketch = Arc<RwLock<Array1<usize>>>;

/// Compute the sketch of node proximity for a directed graph.  
/// Sketch vector of a node is a list of integers obtained by hashing the weighted list of it neighbours (neigbour, weight).  
pub struct NodeSketchAsym {
    /// specific arguments
    params: NodeSketchParams,
    /// Row compressed matrix representing self loop augmented graph i.e initial neighbourhood info
    csrmat: CsMatI<f64, usize>,
    /// the transposed matrix
    csrmat_transposed: CsMatI<f64, usize>,
    /// degrees in and out
    degrees: Vec<Degree>,
    /// The matrix storing all sketches along iterations for neighbours directed toward current node
    sketches_in: Vec<RowSketch>,
    //
    previous_sketches_in: Vec<RowSketch>,
    /// The matrix storing all sketches along iterations for neighbours reached from current node
    sketches_out: Vec<RowSketch>,
    /// sketches_out state at previous iterations
    previous_sketches_out: Vec<RowSketch>,
} // end of struct NodeSketchAsym

impl NodeSketchAsym {
    pub fn new(params: NodeSketchParams, mut trimat: TriMatI<f64, usize>) -> Self {
        //
        log::info!("=======================================================");
        log::info!("allocating NodeSketchAsym : {:?}", params);
        //
        let csrmat = diagonal_augmentation(&mut trimat, 1.0);
        let csrmat_transposed = csrmat.transpose_view().to_owned().to_csr();
        let mut sketches_in = Vec::<RowSketch>::with_capacity(csrmat.rows());
        let mut previous_sketches_in = Vec::<RowSketch>::with_capacity(csrmat.rows());
        let mut sketches_out = Vec::<RowSketch>::with_capacity(csrmat.rows());
        let mut previous_sketches_out = Vec::<RowSketch>::with_capacity(csrmat.rows());
        let degrees = get_csmat_degrees(&csrmat);
        let sketch_size = params.get_sketch_size();
        for _ in 0..csrmat.rows() {
            // incoming edges
            let sketch = Array1::<usize>::zeros(sketch_size);
            sketches_in.push(Arc::new(RwLock::new(sketch)));
            let previous_sketch_in = Array1::<usize>::zeros(sketch_size);
            previous_sketches_in.push(Arc::new(RwLock::new(previous_sketch_in)));
            // outgoing edges
            let sketch = Array1::<usize>::zeros(sketch_size);
            sketches_out.push(Arc::new(RwLock::new(sketch)));
            let previous_sketch_out = Array1::<usize>::zeros(sketch_size);
            previous_sketches_out.push(Arc::new(RwLock::new(previous_sketch_out)));
        }
        NodeSketchAsym {
            params,
            csrmat,
            csrmat_transposed,
            degrees,
            sketches_in,
            previous_sketches_in,
            sketches_out,
            previous_sketches_out,
        }
    } // end of for NodeSketchAsym::new

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
    } // end of get_nb_nodes

    /// We must initialize self.previous_sketches_out
    fn sketch_slamatrix_out(&mut self, parallel: bool) {
        // a closure to work on previous_sketches_out
        let treat_row_out = |row: usize| {
            let mut probminhash3 = ProbMinHash3::<usize, AHasher>::new(self.get_sketch_size(), row);
            let col_range = self.csrmat.indptr().outer_inds_sz(row);
            log::trace!("sketch_slamatrix i : {}, col_range : {:?}", row, col_range);
            for k in col_range {
                let j = self.csrmat.indices()[k];
                let w = self.csrmat.data()[k];
                log::trace!(
                    "sketch_slamatrix row : {}, k  : {}, col {}, w {}",
                    row,
                    k,
                    j,
                    w
                );
                probminhash3.hash_item(j, &w);
            }
            let sketch = probminhash3.get_signature();
            for j in 0..self.get_sketch_size() {
                self.previous_sketches_out[row].write()[j] = sketch[j];
            }
        };

        if !parallel {
            log::debug!(" not parallel case nb rows  {}", self.csrmat.rows());
            for row in 0..self.csrmat.rows() {
                if self.csrmat.indptr().nnz_in_outer_sz(row) > 0 {
                    log::trace!("sketching row {}", row);
                    treat_row_out(row);
                }
            }
        } else {
            // parallel case
            (0..self.csrmat.rows()).into_par_iter().for_each(|row| {
                if self.csrmat.indptr().nnz_in_outer_sz(row) > 0 {
                    treat_row_out(row);
                }
            })
        }
        //
        log::debug!("sketch_slamatrix_out done");
    } // end of sketch_slamatrix_out

    /// We must initialize self.previous_sketches_in
    fn sketch_slamatrix_in(&mut self, parallel: bool) {
        //
        // a closure to treat rows but working on transposed_mat instead of self.csrmat as in sketch_slamatrix_out
        let treat_row_in = |row: usize| {
            let mut probminhash3 = ProbMinHash3::<usize, AHasher>::new(self.get_sketch_size(), row);
            let col_range = self.csrmat_transposed.indptr().outer_inds_sz(row);
            log::trace!("sketch_slamatrix i : {}, col_range : {:?}", row, col_range);
            for k in col_range {
                let j = self.csrmat_transposed.indices()[k];
                let w = self.csrmat_transposed.data()[k];
                log::trace!(
                    "sketch_slamatrix row : {}, k  : {}, col {}, w {}",
                    row,
                    k,
                    j,
                    w
                );
                probminhash3.hash_item(j, &w);
            }
            let sketch = probminhash3.get_signature();
            for j in 0..self.get_sketch_size() {
                self.previous_sketches_in[row].write()[j] = sketch[j];
            }
        };
        if !parallel {
            log::debug!(" not parallel case nb rows  {}", self.csrmat.rows());
            for row in 0..self.csrmat_transposed.rows() {
                if self.csrmat_transposed.indptr().nnz_in_outer_sz(row) > 0 {
                    log::trace!("sketch_slamatrix_in sketching row {}", row);
                    treat_row_in(row);
                }
            }
        } else {
            // parallel case
            (0..self.csrmat_transposed.rows())
                .into_par_iter()
                .for_each(|row| {
                    if self.csrmat_transposed.indptr().nnz_in_outer_sz(row) > 0 {
                        treat_row_in(row);
                    }
                })
        }
        //
        log::debug!("sketch_slamatrix_in done");
    } // end of sketch_sla_matrix_in

    /// We must initialize self.previous_sketches_in and self.previous_sketches_out
    fn sketch_slamatrix(&mut self, parallel: bool) {
        self.sketch_slamatrix_out(parallel);
        self.sketch_slamatrix_in(parallel);
    } // end of sketch_slamatrix

    // do iteration on sketches separately for in neighbours and out neighbours
    fn iteration(&mut self) {
        log::debug!("nodesketchasym : serial_iteration");
        // now we repeatedly merge csrmat (loop augmented matrix) with sketches
        for (row, _) in self.csrmat.outer_iterator().enumerate() {
            // new neighbourhood for current iteration
            self.treat_row_and_col(&row);
        } // end of for on row
          // transfer sketches into previous in sketches
        for i in 0..self.get_nb_nodes() {
            let mut row_write = self.previous_sketches_in[i].write();
            for j in 0..self.get_sketch_size() {
                row_write[j] = self.sketches_in[i].read()[j];
            }
        }
        // transfer sketches into previous out sketches
        for i in 0..self.get_nb_nodes() {
            let mut row_write = self.previous_sketches_out[i].write();
            for j in 0..self.get_sketch_size() {
                row_write[j] = self.sketches_out[i].read()[j];
            }
        }
    } // end of iteration

    // do iteration on sketches separately for in neighbours and out neighbours
    fn parallel_iteration(&mut self) {
        log::trace!("nodesketchasym : parallel_iteration");
        // now we repeatedly merge csrmat (loop augmented matrix) with sketches
        (0..self.csrmat.rows())
            .into_par_iter()
            .for_each(|row| self.treat_row_and_col(&row));
        // transfer sketches into previous sketches
        for i in 0..self.get_nb_nodes() {
            let mut row_write = self.previous_sketches_out[i].write();
            for j in 0..self.get_sketch_size() {
                row_write[j] = self.sketches_out[i].read()[j];
            }
        }
        for i in 0..self.get_nb_nodes() {
            let mut row_write = self.previous_sketches_in[i].write();
            for j in 0..self.get_sketch_size() {
                row_write[j] = self.sketches_in[i].read()[j];
            }
        }
    } // end of parallel_iteration

    // given a row (its number and the data in compressed row Matrice corresponding)
    // the function computes sketch value given previous sketch values
    fn treat_row_and_col(&self, row: &usize) {
        // if we have no data ar row we return immediately
        let row_vec = self.csrmat.outer_view(*row);
        let weight = self.get_decay_weight() / self.get_sketch_size() as f64;
        if row_vec.is_none() {
            log::debug!(
                "NodeSketchAsym::treat_row_and_col, no outgoing edge for row : {}",
                row
            );
        }
        let row_vec = row_vec.unwrap();
        // out treatment new neighbourhood for current iteration
        let mut v_k = HashMap::<usize, f64, ahash::RandomState>::default();
        // get an iterator on neighbours of node corresponding to row
        let row_iter = row_vec.iter();
        for neighbour in row_iter {
            // neighbour.0 is a neighbour of row, it is brought with the weight connection from row to neighbour
            match v_k.get_mut(&neighbour.0) {
                Some(val) => {
                    *val += *neighbour.1;
                    log::trace!(
                        "{} augmenting weight in v_k for neighbour {},  new weight {:.3e}",
                        neighbour.0,
                        *neighbour.1,
                        *val
                    );
                }
                None => {
                    log::trace!(
                        "adding node in v_k {}  weight {:.3e}",
                        neighbour.0,
                        *neighbour.1
                    );
                    v_k.insert(neighbour.0, *neighbour.1);
                }
            };
            // get component due to previous sketch of neighbour
            let neighbour_sketch = &*self.previous_sketches_out[neighbour.0].read();
            for n in neighbour_sketch {
                // something (here n) in a neighbour sketch is brought with the weight connection from row to neighbour multiplied by the decay factor
                match v_k.get_mut(n) {
                    // neighbour sketch contribute with weight neighbour.1 * decay / sketch_size to
                    Some(val) => {
                        *val += weight * *neighbour.1;
                        log::trace!("{} sketch augmenting node {} weight in v_k with decayed edge weight {:.3e} new weight {:.3e}", 
                                    neighbour.0 , *n, weight * *neighbour.1, *val);
                    }
                    None => {
                        log::trace!(
                            "{} sketch adding node with {} decayed weight {:.3e}",
                            neighbour.0,
                            *n,
                            weight * *neighbour.1
                        );
                        v_k.insert(*n, weight * neighbour.1);
                    }
                };
            }
        } // end of while on previous sketches out
          // once we have a new list of (nodes, weight) we sketch it to fill the row of new sketches and to compact list of neighbours
          // as we possibly got more.
        let mut probminhash3a = ProbMinHash3a::<usize, AHasher>::new(self.get_sketch_size(), *row);
        probminhash3a.hash_weigthed_hashmap(&v_k);
        let sketch = Array1::from_vec(probminhash3a.get_signature().clone());
        // save sketches into self sketch
        let mut row_write = self.sketches_out[*row].write();
        for j in 0..self.get_sketch_size() {
            row_write[j] = sketch[j];
        }
        // end treatment outgoing edge
        //
        // now the problem for in treatment is that we should better have a csc mat?
        // For a directed graph we suppose the Csr matrix so that mat[[i,j]] is edge from i to j.
        // So we get a transposed view of self.csrmat
        //
        let row_vec = self.csrmat_transposed.outer_view(*row);
        if row_vec.is_none() {
            log::debug!(
                "NodeSketchAsym::treat_row_and_col, no incoming edge for row : {}",
                row
            );
            return;
        }
        let row_vec = row_vec.unwrap();
        // in treatment new neighbourhood for current iteration
        let mut v_k = HashMap::<usize, f64, ahash::RandomState>::default();
        // get an iterator on neighbours of node corresponding to row
        let row_iter = row_vec.iter();
        for neighbour in row_iter {
            // neighbour.0 is a neighbour of row, it is brought with the weight connection from row to neighbour
            match v_k.get_mut(&neighbour.0) {
                Some(val) => {
                    *val += *neighbour.1;
                    log::trace!(
                        "{} augmenting weight in v_k for neighbour {},  new weight {:.3e}",
                        neighbour.0,
                        *neighbour.1,
                        *val
                    );
                }
                None => {
                    log::trace!(
                        "adding node in v_k {}  weight {:.3e}",
                        neighbour.0,
                        *neighbour.1
                    );
                    v_k.insert(neighbour.0, *neighbour.1);
                }
            };
            // get component due to previous sketch of neighbour
            let neighbour_sketch = &*self.previous_sketches_in[neighbour.0].read();
            for n in neighbour_sketch {
                // something (here n) in a neighbour sketch is brought with the weight connection from row to neighbour multiplied by the decay factor
                match v_k.get_mut(n) {
                    // neighbour sketch contribute with weight neighbour.1 * decay / sketch_size to
                    Some(val) => {
                        *val += weight * *neighbour.1;
                        log::trace!("{} sketch augmenting node {} weight in v_k with decayed edge weight {:.3e} new weight {:.3e}", 
                                neighbour.0 , *n, weight * *neighbour.1, *val);
                    }
                    None => {
                        log::trace!(
                            "{} sketch adding node with {} decayed weight {:.3e}",
                            neighbour.0,
                            *n,
                            weight * *neighbour.1
                        );
                        v_k.insert(*n, weight * neighbour.1);
                    }
                };
            }
        } // end of while on previous sketches out
          // once we have a new list of (nodes, weight) we sketch it to fill the row of new sketches and to compact list of neighbours
          // as we possibly got more.
        let mut probminhash3a = ProbMinHash3a::<usize, AHasher>::new(self.get_sketch_size(), *row);
        probminhash3a.hash_weigthed_hashmap(&v_k);
        let sketch = Array1::from_vec(probminhash3a.get_signature().clone());
        // save sketches into self sketch
        let mut row_write = self.sketches_in[*row].write();
        for j in 0..self.get_sketch_size() {
            row_write[j] = sketch[j];
        }
    } // end of treat_row

    /// computes the embedding.
    pub fn compute_embedded(&mut self) -> Result<EmbeddedAsym<usize>, anyhow::Error> {
        //
        log::debug!("NodeSketchAsym compute_Embedded");
        let cpu_start = ProcessTime::now();
        let sys_start = SystemTime::now();
        // first iteration, we fill previous sketches
        let parallel = self.params.get_parallel();
        log::debug!("NodeSketchAsym parallel mode : {}", parallel);
        self.sketch_slamatrix(parallel);
        for i in 0..self.params.get_nb_iter() {
            log::debug!("compute_embedded , num hop {}", i);
            if parallel {
                self.parallel_iteration();
            } else {
                self.iteration();
            }
        }
        let sys_t: f64 = sys_start.elapsed().unwrap().as_millis() as f64 / 1000.;
        log::info!(
            " Embedded sys time(s) {:.2e} cpu time(s) {:.2e}",
            sys_t,
            cpu_start.elapsed().as_secs()
        );
        println!(
            " Embedded sys time(s) {:.2e} cpu time(s) {:.2e}",
            sys_t,
            cpu_start.elapsed().as_secs()
        );
        // allocate the asymetric Embedded
        let nbnodes = self.sketches_in.len();
        let embedded_dim = self.sketches_in[0].read().len();
        let mut embedded_source = Array2::<usize>::zeros((nbnodes, embedded_dim));
        let mut embedded_target = Array2::<usize>::zeros((nbnodes, embedded_dim));
        for i in 0..nbnodes {
            for j in 0..self.get_sketch_size() {
                embedded_target.row_mut(i)[j] = self.sketches_in[i].read()[j];
                embedded_source.row_mut(i)[j] = self.sketches_out[i].read()[j];
            }
        }
        let embedded = EmbeddedAsym::<usize>::new(
            embedded_source,
            embedded_target,
            Some(self.degrees.clone()),
            crate::embed::tools::jaccard::jaccard_distance,
        );
        //
        Ok(embedded)
    } // end of compute_Embedded
} // end of impl NodeSketchAsym

impl EmbedderT<usize> for NodeSketchAsym {
    type Output = EmbeddedAsym<usize>;
    //
    fn embed(&mut self) -> Result<EmbeddedAsym<usize>, anyhow::Error> {
        let res = self.compute_embedded();
        match res {
            Ok(embeded) => {
                return Ok(embeded);
            }
            Err(err) => {
                return Err(err);
            }
        }
    } // end of embed
} // end of impl<f64> EmbedderT<f64>

//=====================================================================================================

#[cfg(test)]
mod tests {

    //    cargo test --features="openblas-static" validation::link::tests::test_name -- --nocapture
    //    RUST_LOG=graphembed::sketching=TRACE cargo test --features="openblas-static" test_nodesketchasym_wiki -- --nocapture

    use crate::prelude::*;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    use super::*;

    #[test]
    fn test_nodesketchasym_wiki() {
        log_init_test();
        // Nodes: 7115 Edges: 103689
        log::debug!("in nodesketchasym  : test_nodesketchasym_wiki");
        let path = std::path::Path::new(crate::DATADIR).join("wiki-Vote.txt");
        log::info!("\n\n test_nodesketchasym_wiki, loading file {:?}", path);
        let res = csv_to_trimat::<f64>(&path, true, b'\t');
        if res.is_err() {
            log::error!("error : {:?}", res.as_ref().err());
            log::error!("test_nodesketchasym_wiki failed in csv_to_trimat");
            assert_eq!(1, 0);
        }
        let (trimat, node_index) = res.unwrap();
        let sketch_size = 150;
        let decay = 0.1;
        let nb_iter = 2;
        let symetric = false;
        let parallel = true;
        let params = NodeSketchParams {
            sketch_size,
            decay,
            nb_iter,
            symetric,
            parallel,
        };
        // now we embed
        let mut nodesketch = NodeSketchAsym::new(params, trimat);
        let sketch_embedding = Embedding::new(node_index, &mut nodesketch);
        if sketch_embedding.is_err() {
            log::error!("error : {:?}", sketch_embedding.as_ref().err());
            log::error!("test_nodesketchasym_wiki failed in compute_Embedded");
            assert_eq!(1, 0);
        }
        let _embed_res = sketch_embedding.unwrap();
    } // end test_nodesketchasym_wiki
} // end of mod tests
