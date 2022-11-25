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

use num_traits::float::*; 

use std::sync::{Arc};
use parking_lot::{RwLock};

use petgraph::graph::DefaultIx;
use petgraph::csr::{Csr};

use crate::tools::{edge::Edge, degrees::*};


pub struct WeightSplit(f32,f32);

impl Default for WeightSplit {
    fn default() -> Self { WeightSplit(0., 0.)}
}

struct EdgeSplit {
    edge : Edge,
    wsplit : WeightSplit
}

impl EdgeSplit {
    fn new(edge : Edge, wsplit : WeightSplit) -> Self {
        EdgeSplit{edge, wsplit}
    }
}


/// Structure describing how the weight of edges is dispatched onto tis vertices.
struct AlphaR {
    r : Vec<f32>,
    alpha : Vec<EdgeSplit>
}

impl AlphaR {
    fn new(r : Vec<f32>, alpha : Vec<EdgeSplit>) -> Self {
        AlphaR{r,alpha}
    }
} // end of impl AlphaR



/// initialize alpha and r (as defined in paper) by Frank-Wolfe algorithm
/// r is dimensioned to number of vertex
fn get_alpha_r<F:Float>(graph : &Csr<F>)  {
    // how many nodes and edges
    let nb_nodes = graph.node_count();
    let nb_edges = graph.edge_count();
    // we will need to update r with // loops on edges.
    // We bet on low degree versus number of edges so low contention (See hogwild)
    let r : Vec<Arc<RwLock<f32>>>= Vec::with_capacity(nb_nodes);
    let mut alpha : Vec<EdgeSplit> = Vec::with_capacity(nb_edges);
    //
    panic!("not yet implemented");

} // end of get_alpha_r


/// check stability of the vertex list gpart with respect to alfar
fn is_stable<F:Float>(graph :  &Csr<F>, alfar : &AlphaR, gpart: &Vec<DefaultIx>) -> bool {
    panic!("not yet implemented");
    return false;
}  // end is_stable


/// build a tentative decomposition from alphar using the PAVA regression
fn try_decomposition(alphar : &AlphaR) -> Vec<Vec<DefaultIx>> {

    panic!("not yet implemented")
} // end of try_decomposition