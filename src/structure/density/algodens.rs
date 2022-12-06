//! density decomposition algorithm driver

#![allow(unused)]



/// We use Graph representation from petgraph.
/// 
/// For Frank-Wolfe iterations, we need // access to edges.
///      We use sprs graph representation and // access to rows.
/// 




use anyhow::{anyhow};



use std::time::{SystemTime};
use cpu_time::ProcessTime;

use num_traits::{float::*, FromPrimitive};

// synchronization primitive
use std::sync::{Arc};
use parking_lot::{RwLock};
use atomic::{Atomic, Ordering};
use rayon::prelude::*;

use petgraph::graph::{Graph, EdgeReference, NodeIndex, DefaultIx};
use petgraph::{Undirected, visit::*};

// to get sorting with index as result
use indxvec::Vecops;
//
use super::pava::{Point, BlockPoint, IsotonicRegression};

/// describes weight of each node of an edge.
#[derive(Copy,Clone,Debug)]
pub struct WeightSplit(f32,f32);

impl Default for WeightSplit {
    fn default() -> Self { WeightSplit(0., 0.)}
}


/// Structure describing how weight of edges is dispatched to nodes.
#[derive(Copy, Clone)]
struct EdgeSplit<'a, F> {
    edge : EdgeReference<'a, F>,
    wsplit : WeightSplit
}

impl <'a, F> EdgeSplit<'a, F> {
    fn new(edge : EdgeReference<'a,F>, wsplit : WeightSplit) -> Self {
        EdgeSplit{edge, wsplit}
    }
}


/// Structure describing how the weight of edges is dispatched onto tis vertices.
struct AlphaR<'a,F> {
    r : Vec<F>,
    alpha : Vec<EdgeSplit<'a,F>>
}

impl <'a, F> AlphaR<'a, F> {
    fn new(r : Vec<F>, alpha : Vec<EdgeSplit<'a,F>>) -> Self {
        AlphaR{r,alpha}
    }
    
    /// get r field
    pub fn get_r(&self) -> &Vec<F> { &self.r}

    /// get alpha field
    pub fn get_alpha(&self) -> &Vec<EdgeSplit<'a,F>> { &self.alpha }

} // end of impl AlphaR



/// initialize alpha and r (as defined in paper) by Frank-Wolfe algorithm
/// returns (alpha, r) alpha is dimensioned to number of edges, r is dimensioned to number of vertex
fn get_alpha_r<'a, N, F>(graph : &'a Graph<N, F, Undirected>, nbiter : usize) -> AlphaR<'a,F>
    where F : Float + FromPrimitive + std::ops::AddAssign<F> + Sync + Send + std::fmt::Debug {
    //
    log::info!("entering Frank-Wolfe iterations");
    let cpu_start = ProcessTime::now();
    let sys_start = SystemTime::now();
    // how many nodes and edges
    let nb_nodes = graph.node_count();
    let nb_edges = graph.edge_count();
    // we will need to update r with // loops on edges.
    // We bet on low degree versus number of edges so low contention (See hogwild)
    let mut alpha : Vec<Arc<RwLock<EdgeSplit<'a, F>>>> = Vec::with_capacity(nb_edges);
    //
    let edges = graph.edge_references();
    for e in edges {
        let weight = e.weight();
        let split = EdgeSplit::new(e, WeightSplit(weight.to_f32().unwrap()/2., weight.to_f32().unwrap()/2.));
        alpha.push(Arc::new(RwLock::new(split)));
    }
    // now we initialize r to 0
    let r :  Vec<Arc<Atomic<F>>> = (0..nb_nodes).into_iter().map(|_| Arc::new(Atomic::<F>::new(F::zero()))).collect();
    //
    // a function that computes r from alpha after each iteration
    //
    let r_from_alpha = | r :  &Vec<Arc<Atomic<F>>> , alpha : &Vec<Arc<RwLock<EdgeSplit<'a, F>>>>|  {
            // reset r to 0
        (0..nb_nodes).into_par_iter().for_each(|i| {
                r[i].store(F::zero(), Ordering::Relaxed);
        });
            // alpha's load transferred to r
        (0..alpha.len()).into_par_iter().for_each(|i| {
                let alpha_i = alpha[i].read();
                let old_value = r[alpha_i.edge.source().index()].load(Ordering::Relaxed);
                r[alpha_i.edge.source().index()].store(old_value + F::from(alpha_i.wsplit.0).unwrap(), Ordering::Relaxed);
                // process target
                let old_value = r[alpha_i.edge.target().index()].load(Ordering::Relaxed);
                r[alpha_i.edge.target().index()].store(old_value + F::from(alpha_i.wsplit.1).unwrap(), Ordering::Relaxed);
            }
        );
    };
    //
    // We dispatch alpha to r
    r_from_alpha(&r, &alpha);
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
            let source = alpha_i.edge.source();
            let target = alpha_i.edge.target();
            let r_source = r[source.index()].load(Ordering::Relaxed);
            let r_target = r[target.index()].load(Ordering::Relaxed);
            // get edge node with min r. The smaller gets the weight
            if r_source <  r_target  { 
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
    // We do not need locks any more, simplify
    let r_s: Vec<F> = r.iter().map( |v| v.load(Ordering::Relaxed)).collect();
    let alpha_s: Vec<EdgeSplit<'a,F>> = alpha.iter().map(|a| a.read().clone()).collect();
    //
    log::info!("frank_wolfe (fn get_alpha_r) sys time(s) {:.2e} cpu time(s) {:.2e}", 
            sys_start.elapsed().unwrap().as_secs(), cpu_start.elapsed().as_secs());
    //
    return AlphaR::new(r_s, alpha_s);       
} // end of get_alpha_r


/// check stability of a given vertex block with respect to alfar (algo 2 of Danisch paper)
fn is_stable<'a, F:Float + std::fmt::Debug, N>(graph : &'a Graph<N, F, Undirected>, alphar : &'a AlphaR<'a,F>, block : &BlockPoint<'a,F>) -> bool 
    where F : Float + std::ops::DivAssign + std::ops::AddAssign + std::fmt::Debug + Sync + Send ,
          N : Copy {
    //
    let mut alfa_tmp = alphar.get_alpha().clone();
    // TODO try to // this loop?
    let mut ptiter = block.get_point_iter();
    while let Some((&pt, rank_pt)) = ptiter.next() {
        let pt_idx = NodeIndex::new(rank_pt);
        // rank guve us the index in graph
        let mut neighbours = graph.neighbors_undirected(pt_idx);
        // is neighbor in block
        while let Some(neighbor) = neighbours.next() {
            let neighbor_u = neighbor.index();
            if block.is_in_block(neighbor_u) == false {
                // then we get edge corresponding to (pt , neighbor), modify alfa. Cannot fail
                let (edge_idx, direction) = graph.find_edge_undirected(pt_idx, neighbor).unwrap();
                let edge = graph.edge_endpoints(edge_idx).unwrap();
                // we must check for order
                assert!(edge.0 == pt_idx && edge.1 == neighbor);
                alfa_tmp[edge_idx.index()].wsplit.0 = 0.;
                alfa_tmp[edge_idx.index()].wsplit.1 = alfa_tmp[edge_idx.index()].edge.weight().to_f32().unwrap();
            }
        }
    } // end while on point in blocks
    // we must recompute r from alfa_tmp
    let nb_nodes = alphar.get_r().len();
    let mut r :  Vec<Arc<Atomic<F>>> = (0..nb_nodes).into_iter().map(|_| Arc::new(Atomic::<F>::new(F::zero()))).collect();
    (0..alfa_tmp.len()).into_par_iter().for_each(|i| {
            let alpha_i = alfa_tmp[i];
            let old_value = r[alpha_i.edge.source().index()].load(Ordering::Relaxed);
            r[alpha_i.edge.source().index()].store(old_value + F::from(alpha_i.wsplit.0).unwrap(), Ordering::Relaxed);
            // process target
            let old_value = r[alpha_i.edge.target().index()].load(Ordering::Relaxed);
            r[alpha_i.edge.target().index()].store(old_value + F::from(alpha_i.wsplit.1).unwrap(), Ordering::Relaxed);
        }
    );
    // we must check that r is greater on block than outside

    panic!("not yet implemented");
    return false;
}  // end is_stable


/// build a tentative decomposition from alphar using the PAVA regression
fn try_decomposition<'a,F:Float>(alphar : &'a AlphaR<'a,F>) -> Vec<Vec<DefaultIx>> {
    panic!("not yet implemented");
} // end of try_decomposition





#[cfg_attr(doc, katexit::katexit)]
/// computes a decomposition of graph in blocks of vertices of decreasing density. 
/// 
/// The blocks satisfy:
///  - $B_{i} \subset B_{i+1}$ 
///  - $B_{0}=\emptyset , B_{max}=V$ where $V$ is the set of vertices of G.
///  - each block must be stable
 
pub fn approximate_decomposition<'a, N, F>(graph : &'a Graph<N, F, Undirected>) 
        where  F : Float + std::fmt::Debug + std::iter::Sum + FromPrimitive 
                        + std::ops::AddAssign + std::ops::DivAssign + Sync + Send ,
               N : Copy {
    //
    let nbiter = 10;
    let alpha_r = get_alpha_r(graph, nbiter);
    let alpha = alpha_r.get_alpha();
    let r = alpha_r.get_r();
    //
    let mut y : Vec<F> = (0..r.len()).into_iter().map(|_| F::zero()).collect();
    for i in 0..alpha.len() {
        let mut weight = F::zero();
        let node_max = if alpha[i].wsplit.0 > alpha[i].wsplit.1 {
            alpha[i].edge.source().index()
        }
        else {
            alpha[i].edge.target().index()
        };
        y[node_max] += *alpha[i].edge.weight();
    } // end of for i
    // go to PAVA algorithm , the decomposition of y in blocks makes a tentative decomposition 
    // as -r increases , y decreases. We begin algo by densest blocks!
    let points : Vec<Point<F>> = (0..r.len()).into_iter().map(|i| Point::new(-r[i], y[i])).collect();
    let iso_regression = IsotonicRegression::new_descending(&points);
    let res_regr = iso_regression.do_isotonic();
    if res_regr.is_err() {
        log::error!("approximate_decomposition failed in iso_regression regression");
        std::process::exit(1);
    }
    let res = iso_regression.check_blocks();
    // we try to get blocks. Must make union of bi to get increasing sequence of blocks
    let nb_blocks = iso_regression.get_nb_block();
    
    let mut stable_blocks = Vec::<BlockPoint<F>>::new();
    let mut block_start: Option<BlockPoint<F>> = None;
     
    for i in 0..nb_blocks {
        let block = iso_regression.get_block(i).unwrap().clone();
        if block_start == None {
            block_start = Some(block);
        }
        else {
            block_start.as_mut().unwrap().merge(&block);
        }
        // if bi is stable we push it
        if is_stable(graph, &alpha_r, block_start.as_ref().unwrap()) {
            stable_blocks.push(block_start.unwrap());
        } else {
            continue;
        }
    }
    if block_start.is_some() {
        panic!("should not happen");
    }
    // now we have in blockunion, an increasing family of stable blocks
    panic!("is_stable is not finished");
} // end of approximate_decomposition

//==========================================================================================================

mod tests {
    use super::*;

    use crate::io::csv::weighted_csv_to_graphmap;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    } 

    #[test]
    fn density_miserables() {
        log_init_test();
        //
        log::debug!("in algodens density_miserables");
        let path = std::path::Path::new(crate::DATADIR).join("moreno_lesmis").join("out.moreno_lesmis_lesmis");
        log::info!("\n\n algodens::density_miserables, loading file {:?}", path);
        let res = weighted_csv_to_graphmap::<u32, f64, Undirected>(&path, b' ');
        if res.is_err() {
            log::error!("algodens::density_miserables failed in csv_to_trimat");
            assert_eq!(1, 0);
        }
        // now we can convert into a Graph
        let graph = res.unwrap().into_graph::<>();
        // check get_alpha_r
        let alpha_r = get_alpha_r(&graph, 5);
        let r = alpha_r.get_r();
        log::debug!("r : {:?}", &r[0..20]);
        //
        approximate_decomposition(&graph);
    }
} // end of mod tests