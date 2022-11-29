//! density decomposition algorithm driver

//#![allow(unused)]

/// We use Csr graph representation from petgraph.
/// 
/// For Frank-Wolfe iterations, we need // access to edges.
///      We use sprs graph representation and // access to rows.
/// 




use anyhow::{anyhow};



use std::time::{SystemTime};
use cpu_time::ProcessTime;

use num_traits::{float::*, FromPrimitive};

use std::sync::{Arc};
use parking_lot::{RwLock};
use atomic::{Atomic, Ordering};
use rayon::prelude::*;

use petgraph::graph::{DefaultIx};
use petgraph::csr::{Csr, EdgeReference};
use petgraph::{Undirected, visit::*};



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
/// returns (alpha, r) alpha is dimensioned to number of edges, r is dimensioned to number of vertex
fn get_alpha_r<'a, F>(graph : &'a Csr<(), F, Undirected>, nbiter : usize) -> (Vec<Arc<RwLock<EdgeSplit<'a, F, Undirected>>>> , Vec<Arc<Atomic<F>>>)
    where F : Float + FromPrimitive + std::ops::AddAssign<F> + Sync + Send {
    //
    // how many nodes and edges
    let nb_nodes = graph.node_count();
    let nb_edges = graph.edge_count();
    // we will need to update r with // loops on edges.
    // We bet on low degree versus number of edges so low contention (See hogwild)
    let mut alpha : Vec<Arc<RwLock<EdgeSplit<'a, F, Undirected>>>> = Vec::with_capacity(nb_edges);
    //
    let edges = graph.edge_references();
    for e in edges {
        let weight = e.weight();
        let split = EdgeSplit::new(e, WeightSplit(weight.to_f32().unwrap()/2., weight.to_f32().unwrap()/2.));
        alpha.push(Arc::new(RwLock::new(split)));
    }
    // now we initialize r
    let r :  Vec<Arc<Atomic<F>>> = (0..nb_nodes).into_iter().map(|i| Arc::new(Atomic::<F>::new(F::zero()))).collect();
    // function that computes r from alpha
    let r_from_alpha = | r :  &Vec<Arc<Atomic<F>>> , alpha : &Vec<Arc<RwLock<EdgeSplit<'a, F, Undirected>>>>|  {
            (0..alpha.len()).into_par_iter().for_each(|i| {
                let alpha_i = alpha[i].read();
                let old_value = r[alpha_i.edge.source() as usize].load(Ordering::Relaxed);
                r[alpha_i.edge.source() as usize].store(old_value + *alpha_i.edge.weight(), Ordering::Relaxed);
                r[alpha_i.edge.target() as usize].store(old_value + *alpha_i.edge.weight(), Ordering::Relaxed);
            }
        );
    };
    //
    // We dispatch alpha to r
    r_from_alpha(&r, &alpha);
    // we can do iterations
    (0..alpha.len()).into_par_iter().for_each(|i| {
            // treat first node
            let alpha_i = alpha[i].read();
            let weight = F::from(alpha_i.wsplit.0).unwrap();
            let node = alpha_i.edge.source();
            let arc = &r[node as usize];
            let new_value =  arc.load(Ordering::Relaxed) + weight;;
            arc.store(new_value, Ordering::Relaxed);
            // treat second node
            let weight = F::from(alpha_i.wsplit.1).unwrap();
            let node = alpha_i.edge.target();
            let arc = &r[node as usize];
            let new_value =  arc.load(Ordering::Relaxed) + weight;;
            arc.store(new_value, Ordering::Relaxed);
        }
    );
    // now do iterations
    let delta_e : Vec<Arc<RwLock<WeightSplit>>> = (0..nb_edges).into_iter().map(
        |_| Arc::new(RwLock::new(WeightSplit::default()))
    ).collect();
    //
    for iter in 0..nbiter {
        let gamma = 2. / (2. + iter as f32);
        //
        log::info!("iteration : {}, {:.3e}", iter, gamma);
        //
        (0..alpha.len()).into_par_iter().for_each(|i| {
            let mut delta_i = delta_e[i].write();
            let alpha_i = alpha[i].read();
            // get edge node with min r. The smaller gets the weight
            if alpha_i.wsplit.0 <  alpha_i.wsplit.1  { 
                delta_i.0 = alpha_i.edge.weight().to_f32().unwrap();
                delta_i.1 = 0.;

            } else {
                delta_i.1 = alpha_i.edge.weight().to_f32().unwrap();
                delta_i.0 = 0.;
            };
        }); // end of // computation of 
        // update delta_e for evolution of alpha
        (0..alpha.len()).into_par_iter().for_each(|i|  {
            let delta_i = delta_e[i].read();
            let mut alpha_i = alpha[i].write();
            alpha_i.wsplit.0 =  (1. - gamma) * alpha_i.wsplit.0 + gamma * delta_i.0;    
            alpha_i.wsplit.1 =  (1. - gamma) * alpha_i.wsplit.1 + gamma * delta_i.1;    
        }); 
        // now we recompute r
        r_from_alpha(&r, &alpha);
    } // end of // loop on edges 
    //
    return (alpha, r);       
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