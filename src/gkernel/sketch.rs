//! sketching work on mgraph
//! The module is inspired from the paper Global Weisfeiler Lehman Kernel Morris-Kersting 2017 [https://arxiv.org/abs/1703.02379]



// use anyhow::{anyhow};
// use log::log_enabled;

use ndarray::{Array1};

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

use super::mgraph::*;
use super::params::*;

/// to store the sketching result
pub type NodeSketch<Nlabel, Elabel> = Arc<RwLock<Array1<(Nlabel, Elabel)>>>;

pub struct MgraphSketch<'b, 'a, NodeId, Nlabel, Elabel> 
    where NodeId : IdT,
          Nlabel : LabelT,
          Elabel : LabelT {
    /// 
    mgraph : &'b Mgraph<'a, NodeId, Nlabel, Elabel>,
    ///
    nodeindex : IndexSet<NodeId>,
    /// The vector storing node sketch along iterations, length is nbnodes, each RowSketch is a vecotr of sketch_size
    sketches : Vec<NodeSketch<Nlabel, Elabel> >,
    ///
    previous_sketches : Vec<NodeSketch<Nlabel, Elabel>>, 
}  // end of struct MgraphSketch



impl<'b, 'a, NodeId, Nlabel, Elabel> MgraphSketch<'b, 'a, NodeId, Nlabel, Elabel> 
    where   NodeId : IdT,
            Elabel : LabelT,
            Nlabel : LabelT  {

    /// allocation
    pub fn new(mgraph : &'b Mgraph<'a, NodeId, Nlabel, Elabel>, params : SketchParams) -> Self {
        // allocation of nodeindex
        let nb_nodes = mgraph.get_nb_nodes();
        // first initialization of previous sketches
        let mut nodeindex = IndexSet::<NodeId>::with_capacity(nb_nodes);
        let mut iter = mgraph.get_node_iter();
        while let Some(node) = iter.next() {
            nodeindex.insert(*node.0);
        }
        assert_eq!(nb_nodes, nodeindex.len());
        let mut sketches = Vec::<NodeSketch<Nlabel, Elabel>>::with_capacity(nb_nodes);
        let mut previous_sketches = Vec::<NodeSketch<Nlabel, Elabel>>::with_capacity(nb_nodes);    
        //
        for _ in 0..nb_nodes {
            /* 
            let sketch = Array1::<(Nlabel, Elabel)>::zeros(params.get_sketch_size());
            sketches.push(Arc::new(RwLock::new(sketch)));
            let previous_sketch = Array1::<(Nlabel, Elabel)>::zeros(params.get_sketch_size());
            previous_sketches.push(Arc::new(RwLock::new(previous_sketch))); 
            */
        }
        std::process::exit(1);
    } // end of new



    /// updte sketches from previous sketches
    fn update_node(&self, nodeid : &NodeId) {
        // WE MUST NOT FORGET Self Loop Augmentation
        //

    } // end of update_node

    /// parallel iteration on nodes to update sketches
    fn one_iteration(&self) {

    }
}  // end of impl MgraphSketch