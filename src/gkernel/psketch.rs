//! Sketching work on pgraph.  
//! The module is inspired from the paper Global Weisfeiler Lehman Kernel Morris-Kersting 2017 <https://arxiv.org/abs/1703.02379>
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

use ndarray::{Array1};
// use indexmap::IndexSet;
//use std::ops::Index;

//use std::hash::Hash;
// use std::cmp::Eq;

// use std::fmt::Display;

use std::time::{SystemTime};
use cpu_time::ProcessTime;

use petgraph::graph::{Graph, IndexType};
use petgraph::stable_graph::{NodeIndex, DefaultIx};
use petgraph::visit::*;
use petgraph::{EdgeType,Directed, Direction};

use std::collections::HashMap;
use probminhash::probminhasher::*;

use crate::tools::edge::{EdgeDir};

use super::pgraph::*;
use super::params::*;


/// To sketch/store the node sketching result
/// Exploring nodes around a node we skecth the Node labels encountered 
pub type NSketch<Nlabel> = Arc<RwLock<Vec<Nlabel>>>;

/// To sketch/store the edge encountered around a node
/// Exploring nodes around a node we skecth the Edge labels encountered 
/// 
pub type ESketch<Elabel> = Arc<RwLock<Vec<Elabel>>>;

/// We sketch and store the transition using the couple of labels
pub type NESketch<Nlabel, Elabel> = Arc<RwLock<Vec<NElabel<Nlabel, Elabel>>>>;


/// At each hop around a node we register the node label and edge label encountered.
/// So the (node label, edge label) represent the information encountered during the hop
/// We can combine Jaccard distance between the 2 Sketch Vectors.
pub struct Sketch<Nlabel, Elabel> {
    /// 
    sketch_size : u32,
    ///
    n_sketch : NSketch<Nlabel>,
    ///
    e_sketch :  ESketch<Elabel>, 
    ///
    ne_sketch : NESketch<Nlabel, Elabel>,
} // end of struct Sketch


impl<Nlabel,Elabel> Sketch<Nlabel, Elabel> 
    where Nlabel : LabelT,
          Elabel : LabelT {
    ///
    pub fn new(sketch_size : usize) -> Self {
        let nsketch = (0..sketch_size).into_iter().map(|_| Nlabel::default()).collect();
        let esketch = (0..sketch_size).into_iter().map(|_| Elabel::default()).collect();
        let nesketch : Vec<NElabel<Nlabel, Elabel>> = (0..sketch_size).into_iter().map(|_| NElabel::default()).collect();

        Sketch{sketch_size : u32::from_usize(sketch_size).unwrap(), n_sketch : Arc::new(RwLock::new(nsketch)), 
                                        e_sketch: Arc::new(RwLock::new(esketch)),
                                        ne_sketch: Arc::new(RwLock::new(nesketch))}
    }

    /// get a reference on node sketch by Nlabel
    pub fn get_n_sketch(&self) -> &NSketch<Nlabel> {
        &self.n_sketch
    } 

    /// get a reference on node sketch by Elabel
    pub fn get_e_sketch(&self) -> &ESketch<Elabel> {
        &self.e_sketch
    }

    /// get a reference on node sketch by (Nlabel, Elabel)
    pub fn get_ne_sketch(&self) -> &NESketch<Nlabel,Elabel> {
        &self.ne_sketch
    } 


    /// get sketch length
    pub fn get_sketch_size(&self) -> usize {
        self.sketch_size as usize
    }

}  // end of Sketch


/// This structs gathers sketches and ensure transition during iterations.
/// Its index in current_sketch being the index of the node in the graph indexing
/// Vectors are diemnsioned to number of nodes. Sketch is dimensioned to sketch_size
struct SketchTransition<Nlabel, Elabel> {
    nb_nodes : usize,
    //
    nb_sketch : usize,
    /// At a given index we have the sketch of the node of the corresponding index in the graph indexing
    current_sketch : Vec<Sketch<Nlabel, Elabel>>,
    ///
    previous_sketch : Vec<Sketch<Nlabel, Elabel>>,
} // end of SketchTransition


impl <Nlabel, Elabel> SketchTransition<Nlabel, Elabel> 
    where Nlabel : LabelT , Elabel : LabelT {

    pub fn new(nb_nodes : usize, nb_sketch : usize) -> Self {
        let current_sketch : Vec<Sketch<Nlabel, Elabel>> = (0..nb_nodes).into_iter().map(|_|  Sketch::<Nlabel, Elabel>::new(nb_sketch)).collect();
        let previous_sketch : Vec<Sketch<Nlabel, Elabel>> = (0..nb_nodes).into_iter().map(|_|  Sketch::<Nlabel, Elabel>::new(nb_sketch)).collect();
        SketchTransition{nb_nodes, nb_sketch, current_sketch, previous_sketch}
    }

    /// 
    #[allow(unused)]
    pub(crate) fn get_mut_current(&mut self) -> &mut Vec<Sketch<Nlabel, Elabel>> {
        return &mut self.current_sketch;
    }

    ///
    #[allow(unused)]
    pub(crate) fn get_mut_previous(&mut self) ->  &mut Vec<Sketch<Nlabel, Elabel>> {
        return &mut self.previous_sketch;
    }
 
   /// 
   pub(crate) fn get_current(&self) -> &Vec<Sketch<Nlabel, Elabel>> {
        return &self.current_sketch;
    }   

    ///
    pub(crate) fn get_previous(&self) ->  &Vec<Sketch<Nlabel, Elabel>> {
        return &self.previous_sketch;
    }


    /// does the transition between iterations, for all nodes, for sketch based on node labels : transfer current to previous
    /// TODO can be made // , locks are here for that.
    pub(crate) fn transfer_n(&self) {
        for i in 0..self.nb_nodes { 
            let mut row_write = self.previous_sketch[i].n_sketch.write();
            for j in 0..self.nb_sketch {
                row_write[j] = self.current_sketch[i].n_sketch.read()[j].clone();
            }
        }  
    } // end of tranfer_n


    /// does the transition between iterations, for all nodes, for sketch based on couple (node label, edge label) : transfer current to previous
    pub fn transfer_ne(&self) {
        for i in 0..self.nb_nodes { 
            let mut row_write = self.previous_sketch[i].ne_sketch.write();
            for j in 0..self.nb_sketch {
                row_write[j] = self.current_sketch[i].ne_sketch.read()[j].clone();
            }
        }  
    } // end of tranfer_ne 

    pub(crate) fn iteration_transition(&self) {
        self.transfer_n();
        self.transfer_ne();
    } // end of iteration_transition

} // end of impl SketchTransition


// When using graph asymtry we have distinct transitions for IN and OUT edge
struct AsymetricTransition<Nlabel, Elabel>  {
    /// store transition on incoming  edges
    t_in : SketchTransition<Nlabel, Elabel> ,
    /// store transition on outgoing  edges
    t_out : SketchTransition<Nlabel, Elabel> ,
} // end of AsymetricTransition



impl <Nlabel, Elabel> AsymetricTransition<Nlabel, Elabel> 
    where Nlabel : LabelT , Elabel : LabelT {

    pub fn new(nb_nodes : usize, nb_sketch : usize) -> Self {
        // in initialization
        let t_in = SketchTransition::new(nb_nodes, nb_sketch);
        let t_out = SketchTransition::new(nb_nodes, nb_sketch);
        AsymetricTransition{t_in, t_out}
    }

    /// 
    #[allow(unused)]
    pub(crate) fn get_mut_current(&mut self, dir : EdgeDir) -> &mut Vec<Sketch<Nlabel, Elabel>> {
        match dir  {
            EdgeDir::IN => { return &mut self.t_in.current_sketch; },
            EdgeDir::OUT => { return &mut self.t_out.current_sketch; },
            EdgeDir::INOUT => { std::panic!("should not happen")},
        }
    } // end of get_mut_current


    /// return previous sketch in direction dir
    #[allow(unused)]
    pub(crate) fn get_mut_previous(&mut self, dir : EdgeDir) -> &mut Vec<Sketch<Nlabel, Elabel>> {
        match dir  {
            EdgeDir::IN => { return &mut self.t_in.previous_sketch; },
            EdgeDir::OUT => { return &mut self.t_out.previous_sketch; },
            EdgeDir::INOUT => { std::panic!("should not happen")},
        }
    } // end of get_mut_previous


    /// does the transition between iterations, for all nodes, for sketch based on couple (node label, edge label) : transfer current to previous
    pub(crate) fn transfer_ne(&self) {
        self.t_in.transfer_ne();
        self.t_out.transfer_ne();
    } // end of tranfer_ne 



    pub(crate) fn transfer_n(&self) {
        self.t_in.transfer_n();
        self.t_out.transfer_n();
    } // end of tranfer_n 


    pub(crate) fn iteration_transition(&self) {
        self.transfer_n();
        self.transfer_ne();
    } // end of iteration_transition


} // end of AsymetricTransition



//==========================================================================

/// This structure provides sketching for symetric and asymetric labeled graph.  
/// NodeData and EdgeData are Weights attached to Node and Edge in the petgraph terminology.  
/// For our sketching these attached data must satisfy traits (HasNweight)[HasNweight] and (HasEweight)[HasEweight].  
/// Labels can be attributed to node and edges
pub struct MgraphSketch<'a, Nlabel, Elabel, NodeData, EdgeData, Ty = Directed, Ix = DefaultIx> 
    where Nlabel : LabelT,
          Elabel : LabelT, 
          NodeData : HasNweight<Nlabel> + Send + Sync,
          EdgeData : HasEweight<Elabel> + Send + Sync {
    /// 
    graph : &'a mut Graph< NodeData , EdgeData, Ty, Ix>,
    /// sketching parameters
    sk_params : SketchParams,
    /// has single loop augmentation been done ?
    is_sla : bool,
    ///
    symetric_transition : Option<SketchTransition<Nlabel, Elabel>>,
    ///
    asymetric_transition : Option<AsymetricTransition<Nlabel, Elabel>>,
    ///
    parallel : bool, 
}  // end of struct MgraphSketch



impl<'a, Nlabel, Elabel, NodeData, EdgeData, Ty, Ix> MgraphSketch<'a, Nlabel, Elabel, NodeData, EdgeData, Ty, Ix> 
    where   Elabel : LabelT,
            Nlabel : LabelT,
            NodeData : HasNweight<Nlabel> + Send + Sync,
            EdgeData : Default + HasEweight<Elabel> + Send + Sync,
            Ty : EdgeType + Send + Sync,
            Ix : IndexType + Send + Sync  {

    /// allocation
    pub fn new(graph : &'a mut  Graph<NodeData, EdgeData, Ty, Ix>, params : SketchParams) -> Self {
        // allocation of nodeindex
        let nb_nodes = graph.node_count();
        let nb_sketch = params.get_sketch_size();
        // first initialization of previous sketches
        let symetric_sketch : Option<SketchTransition::<Nlabel, Elabel>>;
        let asymetric_transition : Option<AsymetricTransition<Nlabel, Elabel>>;
        if params.is_symetric() {
            symetric_sketch = Some(SketchTransition::<Nlabel, Elabel>::new(nb_nodes, nb_sketch));       
            asymetric_transition = None;
        }
        else {
            symetric_sketch = None;
            asymetric_transition = Some(AsymetricTransition::<Nlabel, Elabel>::new(nb_nodes, nb_sketch));
        }
        //
        MgraphSketch{ graph : graph , sk_params : params, is_sla : false, symetric_transition : symetric_sketch, 
            asymetric_transition, parallel : false}
    } // end of new

    /// check if graph has edge labels to avoid useless computations
    fn graph_has_elabels(&self) -> bool {
        true
    }  // end of graph_has_elabels



    /// get current sketch of node
    pub fn get_current_sketch_node(&self, node : usize, dir : EdgeDir) -> Option<&Sketch<Nlabel, Elabel>> {
        match dir {
            EdgeDir::INOUT => {
                if self.symetric_transition.is_some() {
                    return Some(&self.symetric_transition.as_ref().unwrap().get_current()[node]);
                }
                else {
                    return None;
                }
            },
            EdgeDir::OUT => {
                if self.asymetric_transition.is_some() {
                    return Some(&self.asymetric_transition.as_ref().unwrap().t_out.get_current()[node]);
                }
                else {
                    return None;
                }
            },
            EdgeDir::IN => {
                if self.asymetric_transition.is_some() {
                    return Some(&self.asymetric_transition.as_ref().unwrap().t_in.get_current()[node]);
                }
                else {
                    return None;
                }
            },
        } // end match
    }  // end of get_current_sketch_node



    /// get current sketch of node
    fn get_previous_sketch_node(&self, node : usize,  dir : EdgeDir) -> Option<&Sketch<Nlabel, Elabel>> {
        match dir {
            EdgeDir::INOUT => {
                if self.symetric_transition.is_some() {
                    return Some(&self.symetric_transition.as_ref().unwrap().get_previous()[node]);
                }
                else {
                    return None;
                }
            },
            EdgeDir::OUT => {
                if self.asymetric_transition.is_some() {
                    return Some(&self.asymetric_transition.as_ref().unwrap().t_out.get_previous()[node]);
                }
                else {
                    return None;
                }
            },
            EdgeDir::IN => {
                if self.asymetric_transition.is_some() {
                    return Some(&self.asymetric_transition.as_ref().unwrap().t_in.get_previous()[node]);
                }
                else {
                    return None;
                }
            },
        } // end match
    } // end of get_previous_sketch_node


    /// returns sketch_size 
    pub fn get_sketch_size(&self) -> usize { self.sk_params.get_sketch_size()}


    /// drives the whole computation
    pub fn compute_embedded(&mut self) -> Result<usize,anyhow::Error> {
        log::debug!("in MgraphSketch::compute_Embedded");
        //
        let cpu_start = ProcessTime::now();
        let sys_start = SystemTime::now();
        //
        self.self_loop_augmentation();
        // symetric or asymetric embedding
        let nb_iter = self.sk_params.get_nb_iter();
        // iterations loop
        if self.sk_params.is_symetric() {
            for _ in 0..nb_iter {
                self.one_iteration_symetric();
            }
        } // end symetric case 
        else {
            for _ in 0..nb_iter {
                self.one_iteration_asymetric();
            }           
        }
        //
        let sys_t : f64 = sys_start.elapsed().unwrap().as_millis() as f64 / 1000.;
        println!(" embedding sys time(s) {:.2e} cpu time(s) {:.2e}", sys_t, cpu_start.elapsed().as_secs());
        log::info!(" Embedded sys time(s) {:.2e} cpu time(s) {:.2e}", sys_t, cpu_start.elapsed().as_secs());
        // allocate the (symetric) Embedded
        // TODO must allocate embbedded data. Pb sym/asym type for results. 
        //
        Err(anyhow!("not yet"))
    } // end of compute_embedded


    // TODO This function is the only reason why we need &mut graph in struct!! 
    /// updte sketches from previous sketches
    fn self_loop_augmentation(&mut self) {
        // WE MUST NOT FORGET Self Loop Augmentation
        // loop on all nodeindex
        if !self.is_sla {
        let node_indices = &mut self.graph.node_indices();
            while let Some(idx) = node_indices.next() {
                // TODO must document self loop as default!!!
                self.graph.add_edge(idx, idx, EdgeData::default());
            }
        }
        self.is_sla = true;
    } // end of self_loop_augmentation


    /// returns true if graph is self loop augmented.
    pub fn is_sla(&self) -> bool { self.is_sla}


    /// serial/parallel symetric iteration on nodes to update sketches
    fn one_iteration_symetric(&self) {
        //
        if self.parallel {
            let n_indices : Vec<NodeIndex<Ix>>  = self.graph.node_indices().collect();
            n_indices.into_par_iter().for_each( |ndix| self.treat_node_symetric(&ndix));            
        }
        else {
            self.graph.node_indices().for_each( |ndix| self.treat_node_symetric(&ndix));
        }
        // one all nodes have been treated we must do the current to previous iteration
        self.symetric_transition.as_ref().unwrap().iteration_transition();
    } // end one_iteration_symetric


    // Building block for hash updating. 
    // This function takes a node given by its nodeindex and a direction (Incoming or Outgoing) and process edges in the given direction
    // h_label_n stores hashed values of nodes labels, h_label_e stores labels of edge labels
    fn process_node_edges_labels(&self, ndix : &NodeIndex<Ix>,  dir : Direction, h_label_n : &mut HashMap::<Nlabel, f64, ahash::RandomState>, 
                    h_label_ne : &mut HashMap::<NElabel<Nlabel, Elabel>, f64, ahash::RandomState>) {
        //
        let mut edges = self.graph.edges_directed(*ndix, dir);
        while let Some(edge) = edges.next() {
            // get node and weight attribute, it is brought with the weight connection from row to neighbour
            let e_weight = edge.weight();                           // This is petgraph's weight
            let edge_weight = e_weight.get_eweight();    // This is our Eweight gathering label and f32 weight
            let edge_label = edge_weight.get_label();
            let neighbour_idx = match dir {
                Direction::Outgoing => {edge.target() },
                Direction::Incoming => {edge.source() },
            };
            let n_labels = self.graph[neighbour_idx].get_nweight().get_labels();
            // treatment of h_label_n
            for label in n_labels {
                match h_label_n.get_mut(&label) {
                    Some(val) => {
                        *val = *val + edge_weight.get_weight() as f64;
                        log::trace!("{:?} augmenting weight hashed node labels for neighbour {:?},  new weight {:.3e}", 
                                *ndix, neighbour_idx, *val);  
                    }
                    None => {
                        // we add edge info in h_label_n
                        log::trace!("adding node in hashed node labels {:?}  label : {:?}, weight {:.3e}", neighbour_idx, label, e_weight.get_eweight().get_weight());
                        h_label_n.insert(label.clone(),  edge_weight.get_weight() as f64); 
                    }
                }  // end match
                // treat transition via of couples of labels (node_label , edge_label)
                let ne_label = NElabel(label.clone(), edge_label.clone());
                match h_label_ne.get_mut(&ne_label) {
                    Some(val) => {
                        *val = *val + edge_weight.get_weight() as f64;
                        log::trace!("{:?} augmenting weight hashed node labels for neighbour {:?}, via edge label {:?} ,  new weight {:.3e}", 
                                *ndix, neighbour_idx, *edge_label, *val);  
                    }
                    None => {
                        // we add edge info in h_label_n
                        log::trace!("adding node hashed (node,edge) labels {:?}  n_label : {:?}, e_label : {:?} weight {:.3e}", neighbour_idx, label, *edge_label, edge_weight.get_weight());
                        h_label_ne.insert(ne_label,  edge_weight.get_weight() as f64); 
                    }
                }
            } // end of for on nodes labels          
            // 
            // get component due to previous sketch of current neighbour
            // 
            // we must get node label of neighbour and edge label, first we process nodes labels
            let hop_weight = self.sk_params.get_decay_weight()/self.get_sketch_size() as f64;
            // Problem weight of each label? do we renormalize by number of labels, or the weight of the node
            // will be proportional to the number of its labels??
            let neighbour_sketch = &self.get_previous_sketch_node(neighbour_idx.index(), EdgeDir::INOUT);
            // we take previous sketches and we propagate them to our new Nlabel and Elabel hashmap applying hop_weight
            let neighbour_sketch_n = &*neighbour_sketch.unwrap().get_n_sketch().read();
            for sketch_n in neighbour_sketch_n {
                // something (here sketch_n) in a neighbour sketch is brought with the weight connection from neighbour  ndix to ndix multiplied by the decay factor
                match h_label_n.get_mut(sketch_n) {
                    Some(val)  => {
                        *val = *val + hop_weight * edge_weight.get_weight() as f64;
                        log::trace!("{} sketch augmenting node {} weight in hashmap with decayed edge weight {:.3e} new weight {:.3e}", 
                            neighbour_idx.index(), ndix.index() , hop_weight * edge_weight.get_weight() as f64 ,*val);
                    }
                    _                    => {
                        log::trace!("{} sketch adding n label {:?} with decayed weight {:.3e}", neighbour_idx.index(), sketch_n, 
                                hop_weight * edge_weight.get_weight() as f64);
                                h_label_n.insert(sketch_n.clone(), hop_weight * edge_weight.get_weight() as f64);
                    }
                } // end match
            }
            // now we treat transition via couples (node label, edge label)
            let  neighbour_sketch_ne = &*neighbour_sketch.unwrap().get_ne_sketch().read();
            for sketch_ne in neighbour_sketch_ne {
                match h_label_ne.get_mut(&sketch_ne) {
                    Some(val) => {
                        *val = *val + hop_weight * edge_weight.get_weight() as f64;
                        log::trace!("{:?} augmenting weight in edge hash for neighbour {:?},  new weight {:.3e}", 
                                *ndix, neighbour_idx, *val);  
                    }
                    None => {
                        // we add edge info in h_label_e
                        log::trace!("adding node in hashed edge labels {:?}  label : {:?}, weight {:.3e}", neighbour_idx, edge_label, edge_weight.get_weight());
                        h_label_ne.insert(sketch_ne.clone(),  hop_weight * edge_weight.get_weight() as f64); 
                    }
                }  // end match
            }  // end loop on sketch_ne
        }   // end of while
    } // end of process_node_edges_labels



    // loop on neighbours and sketch
    // We will need two probminhasher : one for Nlabels and one for Elabels
    // In the symetric (undirected) case we must treat both edge target and edge source
    fn treat_node_symetric(&self, ndix : &NodeIndex<Ix>) {
        // ndix should correspond to rank in self.sketches (and so to rank in nodes array in petgraph:::grap
        // self.neighbors_undirected give an iterator on all neighbours
        // we must also get labels of edges
        // Graph:edge_endpoints(e) -> 2 NodeIndex from to
        // Graph.edges_directed(nidx) get an iterator over all edges connected to nidx
        //
        assert!(self.graph.is_directed() == false);
        // TODO we initialize by first label ??
        let mut h_label_n = HashMap::<Nlabel, f64, ahash::RandomState>::default();
        let mut h_label_ne = HashMap::<NElabel<Nlabel, Elabel>, f64, ahash::RandomState>::default();
        //
        self.process_node_edges_labels(ndix, Direction::Outgoing, &mut h_label_n, &mut h_label_ne);
        self.process_node_edges_labels(ndix, Direction::Incoming, &mut h_label_n, &mut h_label_ne);
        // We do probminhash stuff
        let mut probminhash3asha_n = ProbMinHash3aSha::<Nlabel>::new(self.get_sketch_size(), Nlabel::default());
        let mut probminhash3asha_ne = ProbMinHash3aSha::<NElabel<Nlabel, Elabel>>::new(self.get_sketch_size(), NElabel::default());
        probminhash3asha_n.hash_weigthed_hashmap(&h_label_n);
        probminhash3asha_ne.hash_weigthed_hashmap(&h_label_ne);
        // save sketches into self sketch
        // first sketch based on nodes labels, we construct new sketch
        let sketch_n = Array1::from_vec(probminhash3asha_n.get_signature().clone());
        // we set new sketch
        let mut row_write = self.get_current_sketch_node(ndix.index(), EdgeDir::INOUT).unwrap().n_sketch.write();
        for j in 0..self.get_sketch_size() {
            row_write[j] = sketch_n[j].clone();
        }  
        // then we process sketching based on couple (node label, edge label)
        let sketch_ne = Array1::from_vec(probminhash3asha_ne.get_signature().clone());
        let mut row_write = self.get_current_sketch_node(ndix.index(), EdgeDir::INOUT).unwrap().ne_sketch.write();
        for j in 0..self.get_sketch_size() {
            row_write[j] = sketch_ne[j].clone();
        }    
    } // end of treat_node_symetric



    // for asymetric embedding
    fn treat_node_asymetric(&self, ndix : &NodeIndex<Ix>) {
        //
        assert!(self.graph.is_directed() == true);
        //
        // first we treat outgoing edges
        //
        let mut h_label_n = HashMap::<Nlabel, f64, ahash::RandomState>::default();
        let mut h_label_ne = HashMap::<NElabel<Nlabel, Elabel>, f64, ahash::RandomState>::default();
        //
        self.process_node_edges_labels(ndix, Direction::Outgoing, &mut h_label_n,&mut h_label_ne);
        //
        let mut probminhash3asha_n = ProbMinHash3aSha::<Nlabel>::new(self.get_sketch_size(), Nlabel::default());
        let mut probminhash3asha_ne = ProbMinHash3aSha::<NElabel<Nlabel, Elabel>>::new(self.get_sketch_size(), NElabel::default());
        probminhash3asha_n.hash_weigthed_hashmap(&h_label_n);
        probminhash3asha_ne.hash_weigthed_hashmap(&h_label_ne);
        // save sketches into self sketch
        // first sketch based on nodes labels, we construct new sketch
        let sketch_n = Array1::from_vec(probminhash3asha_n.get_signature().clone());
        // we set new sketch
        let mut row_write = self.get_current_sketch_node(ndix.index(), EdgeDir::OUT).unwrap().n_sketch.write();
        for j in 0..self.get_sketch_size() {
            row_write[j] = sketch_n[j].clone();
        } 
        // (node label, edge label) case
        let sketch_ne = Array1::from_vec(probminhash3asha_ne.get_signature().clone());
        // we set new sketch
        let mut row_write = self.get_current_sketch_node(ndix.index(), EdgeDir::OUT).unwrap().ne_sketch.write();
        for j in 0..self.get_sketch_size() {
            row_write[j] = sketch_ne[j].clone();
        }              
        //
        // now we treat incoming edges
        //
        let mut h_label_n = HashMap::<Nlabel, f64, ahash::RandomState>::default();
        let mut h_label_ne = HashMap::<NElabel<Nlabel, Elabel>, f64, ahash::RandomState>::default();
        //
        self.process_node_edges_labels(ndix, Direction::Outgoing, &mut h_label_n, &mut h_label_ne);
        //
        let mut probminhash3asha_n = ProbMinHash3aSha::<Nlabel>::new(self.get_sketch_size(), Nlabel::default());
        let mut probminhash3asha_ne = ProbMinHash3aSha::<NElabel<Nlabel, Elabel>>::new(self.get_sketch_size(), NElabel::default());
        probminhash3asha_n.hash_weigthed_hashmap(&h_label_n);
        probminhash3asha_ne.hash_weigthed_hashmap(&h_label_ne);
        // save sketches into self sketch
        // first sketch based on nodes labels, we construct new sketch
        let sketch_n = Array1::from_vec(probminhash3asha_n.get_signature().clone());
        // we set new sketch
        let mut row_write = self.get_current_sketch_node(ndix.index(), EdgeDir::OUT).unwrap().n_sketch.write();
        for j in 0..self.get_sketch_size() {
            row_write[j] = sketch_n[j].clone();
        } 
        // (node label, edge label) case
        let sketch_ne = Array1::from_vec(probminhash3asha_ne.get_signature().clone());
        // we set new sketch
        let mut row_write = self.get_current_sketch_node(ndix.index(), EdgeDir::OUT).unwrap().ne_sketch.write();
        for j in 0..self.get_sketch_size() {
            row_write[j] = sketch_ne[j].clone();
        }    
    } // end of treat_node_asymetric



    /// serial/parallel symetric iteration on nodes to update sketches
    fn one_iteration_asymetric(&self) {
        //
        if self.parallel {
            let n_indices : Vec<NodeIndex<Ix>>  = self.graph.node_indices().collect();
            n_indices.into_par_iter().for_each( |ndix| self.treat_node_asymetric(&ndix));            
        }
        else {
            self.graph.node_indices().for_each( |ndix| self.treat_node_asymetric(&ndix));
        }
        // one all nodes have been treated we must do the current to previous iteration
        self.asymetric_transition.as_ref().unwrap().iteration_transition();
    } // end one_iteration_asymetric

}  // end of impl MgraphSketch

//==============================================================================================


#[cfg(test)]
mod tests {



use super::*; 

use crate::gkernel::exio::maileu::*;

const MAILEU_DIR:&str = "/home/jpboth/Data/Graphs/Mail-EU";

fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}


// a test for data file maileu  Mail-Eu labeled graph <https://snap.stanford.edu/data/email-Eu-core.html>

use super::*;

#[test]
fn test_pgraph_maileu() {
    log_init_test();
    let res_graph = read_maileu_data(String::from(MAILEU_DIR));
    assert!(res_graph.is_ok());
    let mut graph = res_graph.unwrap();
    //
    let skparams = SketchParams::new(100, 0.1, 10, false, false);
    let skgraph = MgraphSketch::new(&mut graph, skparams);
}  // end of test_pgraph_maileu


}  // end of mod tests