//! sketching work on pgraph
//! The module is inspired from the paper Global Weisfeiler Lehman Kernel Morris-Kersting 2017 [https://arxiv.org/abs/1703.02379]
//! 
//! The colouring scheme is replaced by hashing Node Labels and Edge Labels encountered in multihop exploration
//! around a node.
//! We should get both an embedding of each node in terms of Nlabel, Elabel and a global graph summary vector  



use anyhow::{anyhow};
// use log::log_enabled;

use num_traits::cast::FromPrimitive;

//
use parking_lot::{RwLock};
use std::sync::Arc;
use rayon::iter::{ParallelIterator,IntoParallelIterator};

// use indexmap::IndexSet;
//use std::ops::Index;

//use std::hash::Hash;
// use std::cmp::Eq;

// use std::fmt::Display;
// use std::time::{SystemTime};
// use cpu_time::ProcessTime;

use petgraph::graph::{Graph, IndexType, Node};
use petgraph::stable_graph::{NodeIndex, DefaultIx};
use petgraph::visit::*;
use petgraph::{EdgeType,Directed, Direction, Undirected};

use std::collections::HashMap;
use probminhash::probminhasher::*;


use super::pgraph::*;
use super::params::*;


/// To sketch/store the node sketching result
/// Exploring nodes around a node we skecth the Node labels encountered 
pub type NSketch<Nlabel> = Arc<RwLock<Vec<Nlabel>>>;

/// To sketch/store the edge encountered around a node
/// Exploring nodes around a node we skecth the Edge labels encountered 
/// 
pub type ESketch<Elabel> = Arc<RwLock<Vec<Elabel>>>;

/// At each hop around a node we register the node label and edge label encountered.
/// So the (node label, edge label) represent the information encountered during the hop
/// We can combine Jaccard distance between the 2 Sketch Vectors.
pub struct Sketch<Nlabel, Elabel> {
    /// 
    sketch_size : u32,
    ///
    n_sketch : NSketch<Nlabel>,
    ///
    e_sketch :  ESketch<Elabel> 
}


impl<Nlabel,Elabel> Sketch<Nlabel, Elabel> 
    where Nlabel : LabelT,
          Elabel : LabelT {
    ///
    pub fn new(sketch_size : usize) -> Self {
        let nsketch = (0..sketch_size).into_iter().map(|_| Nlabel::default()).collect();
        let esketch = (0..sketch_size).into_iter().map(|_| Elabel::default()).collect();
        Sketch{sketch_size : u32::from_usize(sketch_size).unwrap(), n_sketch : Arc::new(RwLock::new(nsketch)), e_sketch: Arc::new(RwLock::new(esketch))}
    }

    /// get a reference on node sketch by Nlabel
    pub fn get_n_sketch(&self) -> &NSketch<Nlabel> {
        &self.n_sketch
    } 

    /// get a reference on node sketch by Elabel
    pub fn get_e_sketch(&self) -> &ESketch<Elabel> {
        &self.e_sketch
    }

    /// get sketch length
    pub fn get_sketch_size(&self) -> usize {
        self.sketch_size as usize
    }

}  // end of Sketch



pub struct MgraphSketch<'a, Nlabel, Elabel, Ty = Directed, Ix = DefaultIx> 
    where Nlabel : LabelT,
          Elabel : LabelT {
    /// 
    graph : &'a mut Graph<Nweight<Nlabel> , Eweight<Elabel>, Ty, Ix>,
    /// sketching parameters
    sk_params : SketchParams,
    ///
    is_sla : bool,
    /// The vector storing node sketch along iterations, length is nbnodes, each RowSketch is a vector of sketch_size
    /// Its index in current_sketch being the index of the node in the graph indexing
    current_sketch : Vec<Sketch<Nlabel, Elabel>>,
    ///
    previous_sketch : Vec<Sketch<Nlabel, Elabel>>,
    ///
    parallel : bool, 
}  // end of struct MgraphSketch



impl<'a, Nlabel, Elabel, Ty, Ix> MgraphSketch<'a, Nlabel, Elabel, Ty, Ix> 
    where   Elabel : LabelT,
            Nlabel : LabelT,
            Ty : EdgeType + Send + Sync,
            Ix : IndexType + Send + Sync  {

    /// allocation
    pub fn new(graph : &'a mut Graph<Nweight<Nlabel> , Eweight<Elabel>, Ty, Ix>, params : SketchParams) -> Self {
        // allocation of nodeindex
        let nb_nodes = graph.node_count();
        let nb_sketch = params.get_sketch_size();
        // first initialization of previous sketches
        let sketch : Vec<Sketch<Nlabel, Elabel>> = (0..nb_nodes).into_iter().map(|_|  Sketch::<Nlabel, Elabel>::new(nb_sketch)).collect();
        let previous_sketch : Vec<Sketch<Nlabel, Elabel>> = (0..nb_nodes).into_iter().map(|_|  Sketch::<Nlabel, Elabel>::new(nb_sketch)).collect();
        //
        MgraphSketch{ graph : graph , sk_params : params, is_sla : false, current_sketch : sketch, previous_sketch:previous_sketch , parallel : false}
    } // end of new


    /// get current sketch of node
    pub fn get_current_sketch(&self, node : usize) -> &Sketch<Nlabel, Elabel> {
        &self.current_sketch[node]
    }

   /// get current sketch of node
    pub fn get_previous_sketch(&self, node : usize) -> &Sketch<Nlabel, Elabel> {
        &self.previous_sketch[node]
    }

    /// returns sketch_size 
    pub fn get_sketch_size(&self) -> usize { self.sk_params.get_sketch_size()}

    /// drives the whole computation
    pub fn compute_embedded(&mut self) -> Result<usize,anyhow::Error> {
        self.self_loop_augmentation();
        Err(anyhow!("not yet"))
    } // end of compute_embedded



    /// updte sketches from previous sketches
    fn self_loop_augmentation(&mut self) {
        // WE MUST NOT FORGET Self Loop Augmentation
        // loop on all nodeindex
        let node_indices = &mut self.graph.node_indices();
        while let Some(idx) = node_indices.next() {
            self.graph.add_edge(idx, idx, Eweight::<Elabel>::new(Elabel::default(), 1.));
        }
        self.is_sla = true;
    } // end of self_loop_augmentation

    /// returns true if graph is self loop augmented.
    pub fn is_sla(&self) -> bool { self.is_sla}


    /// serial symetric iteration on nodes to update sketches
    fn one_iteration_symetric(&self) {
        //
        if self.parallel {
            let n_indices : Vec<NodeIndex<Ix>>  = self.graph.node_indices().collect();
            n_indices.into_par_iter().for_each( |ndix| self.treat_node_symetric(&ndix));            
        }
        else {
            self.graph.node_indices().for_each( |ndix| self.treat_node_symetric(&ndix));
        }
    } // end one_iteration_symetric


    // Building block for hash updating. 
    // This function takes a node given by its nodeindex and a direction (Incoming or Outgoing) and process edges in the given direction
    // h_label_n stores hashed values of nodes labels, h_label_e stores labels of edge labels
    fn process_node_edges_labels(&self, ndix : &NodeIndex<Ix>,  dir : Direction, h_label_n : &mut HashMap::<Nlabel, f64, ahash::RandomState>, 
            h_label_e : &mut HashMap::<Elabel, f64, ahash::RandomState>) {
        //
        let mut edges = self.graph.edges_directed(*ndix, dir);
        while let Some(edge) = edges.next() {
            // get node and weight attribute, it is brought with the weight connection from row to neighbour
            let e_weight = edge.weight();
            let edge_weight = e_weight.get_weight() as f64;   // This is our weight not petgraph's
            let edge_label = e_weight.get_label();
            let neighbour_idx = match dir {
                Direction::Outgoing => {edge.target() },
                Direction::Incoming => {edge.source() },
            };
            let n_labels = self.graph[neighbour_idx].get_labels();
            // treatment of h_label_n
            for label in n_labels {
                match h_label_n.get_mut(&label) {
                    Some(val) => {
                        *val = *val + edge_weight;
                        log::trace!("{:?} augmenting weight hashed node labels for neighbour {:?},  new weight {:.3e}", 
                                *ndix, neighbour_idx, *val);  
                    }
                    None => {
                        // we add edge info in h_label_n
                        log::trace!("adding node in hashed node labels {:?}  label : {:?}, weight {:.3e}", neighbour_idx, label, e_weight.get_weight());
                        h_label_n.insert(label.clone(),  e_weight.get_weight() as f64); 
                    }
                }  // end match
            }
            // get this edge label
            match h_label_e.get_mut(&edge_label) {
                Some(val) => {
                    *val = *val + edge_weight;
                    log::trace!("{:?} augmenting weight in hashed edge labels for neighbour {:?},  new weight {:.3e}", *ndix, neighbour_idx, *val);  
                }
                None => {
                    // we add edge info in h_label_n
                    log::trace!("adding node in  {:?} , label : {:?},  weight {:.3e}", neighbour_idx, edge_label, e_weight.get_weight());
                    h_label_e.insert(edge_label.clone(),  e_weight.get_weight() as f64); 
                }
            }  // end match            
            // 
            // get component due to previous sketch of current neighbour
            // we must get node label of neighbour and edge label, first we process nodes labels
            let hop_weight = self.sk_params.get_decay_weight()/self.get_sketch_size() as f64;
            // Problem weight of each label? do we renormalize by number of labels, or the weight of the node
            // will be proportional to the number of its labels??
            let neighbour_sketch = &self.previous_sketch[neighbour_idx.index()];
            // we take previous sketches and we propagate them to our new Nlabel and Elabel hashmap applying hop_weight
            let neighbour_sketch = &*neighbour_sketch.get_n_sketch().read();
            for sketch_n in neighbour_sketch {
                // something (here sketch_n) in a neighbour sketch is brought with the weight connection from neighbour  ndix to ndix multiplied by the decay factor
                match h_label_n.get_mut(sketch_n) {
                    Some(val)  => {
                        *val = *val + hop_weight * edge_weight;
                        log::trace!("{} sketch augmenting node {} weight in hashmap with decayed edge weight {:.3e} new weight {:.3e}", 
                            neighbour_idx.index(), ndix.index() , hop_weight * edge_weight ,*val);
                    }
                    _                    => {
                        log::trace!("{} sketch adding n label {:?} with decayed weight {:.3e}", neighbour_idx.index(), sketch_n, hop_weight * edge_weight);
                        h_label_n.insert(sketch_n.clone(), hop_weight * edge_weight);
                    }
                } // end match
            }
            // now we must process edge label
            match h_label_e.get_mut(&edge_label) {
                Some(val) => {
                    *val = *val + edge_weight;
                    log::trace!("{:?} augmenting weight in edge hash for neighbour {:?},  new weight {:.3e}", 
                            *ndix, neighbour_idx, *val);  
                }
                None => {
                    // we add edge info in h_label_e
                    log::trace!("adding node in hashed edge labels {:?}  label : {:?}, weight {:.3e}", neighbour_idx, edge_label, e_weight.get_weight());
                    h_label_e.insert(edge_label.clone(),  edge_weight); 
                }
            }  // end match
        }
    } // end of process_node_edges_labels



    // loop on neighbours and sketch
    // We will need two probminhasher : one for Nlabels and one for Elabels
    // In the symetric (undirected) case we must treat both edge target and edge source
    // We will need two probminhasher : one for Nlabels and one for Elabels

    fn treat_node_symetric(&self, ndix : &NodeIndex<Ix>) {
        // ndix should correspond to rank in self.sketches (and so to rank in nodes array in petgraph:::grap
        // self.neighbors_undirected give an iterator on all neighbours
        // we must also get labels of edges
        // Graph:edge_endpoints(e) -> 2 NodeIndex from to
        // Edge.source() Edge.target() to get nodes extremities
        // Graph.edges_directed(nidx) get an iterator over all edges connected to nidx
        //
        assert!(self.graph.is_directed() == false);
        // TODO we initialize by first label ??
        let mut h_label_n = HashMap::<Nlabel, f64, ahash::RandomState>::default();
        let mut h_label_e = HashMap::<Elabel, f64, ahash::RandomState>::default();
        //
        self.process_node_edges_labels(ndix, Direction::Outgoing, &mut h_label_n, &mut h_label_e);
        self.process_node_edges_labels(ndix, Direction::Incoming, &mut h_label_n, &mut h_label_e);
        // We do probminhash stuff
        let mut probminhash3a_n = ProbMinHash3aSha::<Nlabel>::new(self.get_sketch_size(), Nlabel::default());
        let mut probminhash3a_e = ProbMinHash3aSha::<Elabel>::new(self.get_sketch_size(), Elabel::default());
        probminhash3a_n.hash_weigthed_hashmap(&h_label_n);
        probminhash3a_e.hash_weigthed_hashmap(&h_label_e);
    } // end of treat_node_symetric



}  // end of impl MgraphSketch