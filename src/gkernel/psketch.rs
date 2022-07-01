//! sketching work on pgraph
//! The module is inspired from the paper Global Weisfeiler Lehman Kernel Morris-Kersting 2017 [https://arxiv.org/abs/1703.02379]



// use anyhow::{anyhow};
// use log::log_enabled;


// use ahash::{AHasher};
// use probminhash::probminhasher::*;
//
// use rayon::iter::{ParallelIterator,IntoParallelIterator};
use parking_lot::{RwLock};
use std::sync::Arc;
use indexmap::IndexSet;

//use std::hash::Hash;
// use std::cmp::Eq;

// use std::fmt::Display;
// use std::time::{SystemTime};
// use cpu_time::ProcessTime;

use petgraph::graph::*;
use petgraph::{EdgeType,Directed, Undirected};

use super::pgraph::*;
use super::params::*;


/// to store the node sketching result
/// Exploring nodes around a node we skecth the Node labels encountered 
pub type NSketch<Nlabel> = Arc<RwLock<Vec<Nlabel>>>;

///  to store the edge encountered around a node
/// Exploring nodes around a node we skecth the Edge labels encountered 
/// 
pub type ESketch<Elabel> = Arc<RwLock<Vec<Elabel>>>;

/// At each hop around a node we register the node label and edge label encountered.
/// So the (node label, edge label) represent the information encountered during the hop
/// We can combine Jaccard distance between the 2 Sketch Vectors.
pub struct Sketch<Nlabel, Elabel> {
    sketch_size : usize,
    ///
    n_sketch : NSketch<Nlabel>,
    ///
    e_sketch :  ESketch<Elabel> 
}


impl<Nlabel,Elabel> Sketch<Nlabel, Elabel> 
    where Nlabel : LabelT,
          Elabel : LabelT {
    pub fn new(sketch_size : usize) -> Self {
        let nsketch = (0..sketch_size).into_iter().map(|_| Nlabel::default()).collect();
        let esketch = (0..sketch_size).into_iter().map(|_| Elabel::default()).collect();
        Sketch{sketch_size, n_sketch : Arc::new(RwLock::new(nsketch)), e_sketch: Arc::new(RwLock::new(esketch))}
    }
}  // end of Sketch



pub struct MgraphSketch<'a, Nlabel, Elabel, Ty = Directed, Ix = DefaultIx> 
    where Nlabel : LabelT,
          Elabel : LabelT {
    /// 
    graph : &'a Graph<Nweight<Nlabel> , Eweight<Elabel>, Ty, Ix>,
    /// The vector storing node sketch along iterations, length is nbnodes, each RowSketch is a vecotr of sketch_size
    current_sketch : Vec<Sketch<Nlabel, Elabel>>,
    ///
    previous_sketch : Vec<Sketch<Nlabel, Elabel>>, 
}  // end of struct MgraphSketch



impl<'a, Nlabel, Elabel, Ty, Ix> MgraphSketch<'a, Nlabel, Elabel, Ty, Ix> 
    where   Elabel : LabelT,
            Nlabel : LabelT,
            Ty : EdgeType,
            Ix : IndexType   {

    /// allocation
    pub fn new(graph : &'a Graph<Nweight<Nlabel> , Eweight<Elabel>, Ty, Ix>, params : SketchParams) -> Self {
        // allocation of nodeindex
        let nb_nodes = graph.node_count();
        let nb_sketch = params.get_sketch_size();
        // first initialization of previous sketches
        let sketch : Vec<Sketch<Nlabel, Elabel>> = (0..nb_nodes).into_iter().map(|_|  Sketch::<Nlabel, Elabel>::new(nb_sketch)).collect();
        let previous_sketch : Vec<Sketch<Nlabel, Elabel>> = (0..nb_nodes).into_iter().map(|_|  Sketch::<Nlabel, Elabel>::new(nb_sketch)).collect();
        //
        MgraphSketch{ graph , current_sketch : sketch, previous_sketch:previous_sketch }
    } // end of new



    /// updte sketches from previous sketches
    fn self_loop_augmentation(&mut self) {
        // WE MUST NOT FORGET Self Loop Augmentation
        //

    } // end of self_loop_augmentation

    /// parallel iteration on nodes to update sketches
    fn one_iteration(&self) {

    }
}  // end of impl MgraphSketch