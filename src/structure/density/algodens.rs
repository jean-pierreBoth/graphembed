//! density decomposition algorithm driver

#![allow(unused)]

/// We use Csr graph representation from petgraph.
/// 
/// For Frank-Wolfe iterations, we need // access to edges.
///      We use sprs graph representation and // access to rows.
/// 




use anyhow::{anyhow};



use std::time::{SystemTime};
use cpu_time::ProcessTime;

use num_traits::{float::*, FromPrimitive, ToPrimitive};

use std::sync::{Arc};
use atomic::{Atomic};

use petgraph::graph::{DefaultIx};
use petgraph::csr::{Csr, EdgeReference};
use petgraph::{Undirected, visit::*};

use crate::tools::{degrees::*};


/// describes weight of each node of an edge.
/// Is in accordance with [Edge](Edge), first item correspond to first item of edge.

#[derive(Copy,Clone,Debug)]
pub struct WeightSplit(f32,f32);

impl Default for WeightSplit {
    fn default() -> Self { WeightSplit(0., 0.)}
}


/// Structure describing how weight of edges is dispatched to nodes.
struct EdgeSplit<'a, F, Ty> {
    edge : EdgeReference<'a, F, Ty>,
    wsplit : WeightSplit
}

impl <'a, F, Ty> EdgeSplit<'a, F, Ty> {
    fn new(edge : EdgeReference<'a,F, Ty>, wsplit : WeightSplit) -> Self {
        EdgeSplit{edge, wsplit}
    }
}


/// Structure describing how the weight of edges is dispatched onto tis vertices.
struct AlphaR<'a,F, Ty> {
    r : Vec<F>,
    alpha : Vec<EdgeSplit<'a,F, Ty>>
}

impl <'a, F, Ty> AlphaR<'a, F , Ty> {
    fn new(r : Vec<F>, alpha : Vec<EdgeSplit<'a,F, Ty>>) -> Self {
        AlphaR{r,alpha}
    }
} // end of impl AlphaR



/// initialize alpha and r (as defined in paper) by Frank-Wolfe algorithm
/// r is dimensioned to number of vertex
fn get_alpha_r<'a, F>(graph : &'a Csr<(), F, Undirected>) 
    where F : Float + FromPrimitive {
    // how many nodes and edges
    let nb_nodes = graph.node_count();
    let nb_edges = graph.edge_count();
    // we will need to update r with // loops on edges.
    // We bet on low degree versus number of edges so low contention (See hogwild)
    let r : Vec<Arc<Atomic<F>>> = Vec::with_capacity(nb_nodes);
    let mut alpha : Vec<EdgeSplit<'a, F, Undirected>> = Vec::with_capacity(nb_edges);
    //
    let edges = graph.edge_references();
    for e in edges {
        let weight = e.weight();
       let split = EdgeSplit::new(e, WeightSplit(weight.to_f32().unwrap()/2., weight.to_f32().unwrap()/2.));
       alpha.push(split);
    }
    // now we initialize r
    let r :  Vec<Arc<Atomic<F>>> = (0..nb_nodes).into_iter().map(|i| Arc::new(Atomic::<F>::new(F::zero()))).collect();
    // We dispatch alpha to r
    panic!("not yet implemented");

} // end of get_alpha_r


/// check stability of the vertex list gpart with respect to alfar
fn is_stable<'a, F:Float, Ty>(graph : &'a Csr<F>, alfar : &'a AlphaR<'a,F, Ty>, gpart: &Vec<DefaultIx>) -> bool {
    panic!("not yet implemented");
    return false;
}  // end is_stable


/// build a tentative decomposition from alphar using the PAVA regression
fn try_decomposition<'a,F:Float, Ty>(alphar : &'a AlphaR<'a,F, Ty>) -> Vec<Vec<DefaultIx>> {

    panic!("not yet implemented");
} // end of try_decomposition