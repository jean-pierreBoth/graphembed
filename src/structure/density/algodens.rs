//! density decomposition algorithm driver

//#![allow(unused)]



/// We use Graph representation from petgraph.
/// 
/// For Frank-Wolfe iterations, we need // access to edges.
///      We use sprs graph representation and // access to rows.
/// 






use anyhow::{anyhow};
use log::{log_enabled, Level};

use std::time::{SystemTime};
use cpu_time::ProcessTime;

use num_traits::{float::*, FromPrimitive};

// synchronization primitive
use std::sync::{Arc};
use parking_lot::{RwLock};
use atomic::{Atomic, Ordering};
use rayon::prelude::*;

use indxvec::Vecops;

use petgraph::graph::{Graph, EdgeReference, NodeIndex, DefaultIx};
use petgraph::{Undirected, visit::*};

// to get sorting with index as result
//
use super::pava::{Point, PointBlockLocator, IsotonicRegression, get_point_blocnum};

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
        log::trace!("iteration : {}, {:.3e}", iter, gamma);
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
fn check_stability<'a, F:Float + std::fmt::Debug, N>(graph : &'a Graph<N, F, Undirected>, alphar : &'a AlphaR<'a,F>, 
                    iso_regression : &'a IsotonicRegression<F>) -> StableDecomposition
    where F : Float + std::iter::Sum + FromPrimitive + std::ops::DivAssign + std::ops::AddAssign + std::ops::SubAssign + std::fmt::Debug + Sync + Send ,
          N : Copy {
    //
    let nb_reg_blocks = iso_regression.get_nb_block();
    let pointblocklocator = PointBlockLocator::new(&iso_regression);
    //
    let alfa_tmp = alphar.get_alpha().clone();
    let mut r = alphar.get_r().clone();
    let mut r_test = alphar.get_r().clone();
    // initialize stable_numblocks to sthing  impossible so we know if some points leap through a hole of the algo
    let mut stable_numblocks: Vec<u32> = (0..r.len()).into_iter().map(|_| (nb_reg_blocks+1) as u32).collect();
    let mut points_waiting = Vec::<usize>::with_capacity(r.len());
    let mut block_waiting : u32 = 0;
    //
    for numbloc in 0..nb_reg_blocks {
        log::debug!("\n stability check for block : {}", numbloc);
        let block = iso_regression.get_block(numbloc).unwrap().clone();
        // TODO this iteration can be made // if necessary
        let mut ptiter = block.get_point_iter();
        while let Some((&_pt, rank_pt)) = ptiter.next() {
            let pt_idx = NodeIndex::new(rank_pt);
            points_waiting.push(pt_idx.index());
            // rank guve us the index in graph
            let mut neighbours = graph.neighbors(pt_idx).detach();
            // is neighbor in block
            while let Some((edge_idx,neighbor)) = neighbours.next(graph) {
                let neighbor_u = neighbor.index();
                if pointblocklocator.get_point_block_num(neighbor_u).unwrap() == numbloc+1 {
                    // then we get edge corresponding to (pt , neighbor), modify alfa. Cannot fail
                    let edge = graph.edge_endpoints(edge_idx).unwrap();
                    // we must check for order. We have the same order of of the 2-uple in wsplit and in edge
                    if edge.0 == pt_idx && edge.1 == neighbor {
                        r_test[edge.0.index()] -= F::from(alfa_tmp[edge_idx.index()].wsplit.0).unwrap();
                        r_test[edge.1.index()] += F::from(alfa_tmp[edge_idx.index()].wsplit.0).unwrap();
                    }
                    else if edge.0 == neighbor && edge.1 == pt_idx {
                        r_test[edge.1.index()] -= F::from(alfa_tmp[edge_idx.index()].wsplit.0).unwrap();
                        r_test[edge.0.index()] += F::from(alfa_tmp[edge_idx.index()].wsplit.0).unwrap();                                            
                    }
                    else {
                        panic!("should not happen");
                    }
                }
            }
        } // end while on point in blocks
        // we must check that r is greater on block than outside
        let mut min_in_block = F::max_value();
        let mut max_not_in_block = F::zero();
        (0..r_test.len()).into_iter().for_each(|i| {
            let b = pointblocklocator.get_point_block_num(i).unwrap();
            if b == numbloc {
                min_in_block = min_in_block.min(r_test[i]);
            }
            else if b > numbloc {
                // we 
                max_not_in_block = max_not_in_block.max(r_test[i]);
            }
        }
        );
        log::trace!("stability result  bloc : {}, min in block {:?}, max out block {:?}", numbloc, min_in_block, max_not_in_block);
        if min_in_block > max_not_in_block {
            // got a stable block, we reset r with r_test
            log::info!("stable bloc : {}, regr block : {:?}, min in block {:?}, max out block {:?}",block_waiting, numbloc, min_in_block, max_not_in_block);
            (0..r_test.len()).into_iter().for_each(|i| r[i] = r_test[i]);
            for p in &points_waiting {
                stable_numblocks[*p] = block_waiting;
            }
            block_waiting += 1;
            points_waiting.clear();
            if numbloc >= nb_reg_blocks - 1 {
                log::debug! ("check stability examined all initial regreesion blocks");
                break;
            }
        }
        else {
            // reset r_test to last stable state
            (0..r_test.len()).into_iter().for_each(|i| r_test[i] = r[i]);
        }
    }  // end of loop on initial_blocks
    // a check
    for i in 0..r.len() {
        if stable_numblocks[i] >= (nb_reg_blocks+1) as u32 {
            log::error!(" point is not affected a good block, point : {}, stable block : {}", i, stable_numblocks[i]);
            std::panic!();
        }
    }
    // dump stable_numblocks
    for p in 0..stable_numblocks.len() {
        log::info!("point : {},  bloc : {}", p , stable_numblocks[p]);
    }
    // now we can return stable_numblocks
    StableDecomposition::new(stable_numblocks)
}  // end check_stability


/// build a tentative decomposition from alphar using the PAVA regression
#[allow(unused)]
fn try_decomposition<'a,F:Float>(alphar : &'a AlphaR<'a,F>) -> Vec<Vec<DefaultIx>> {
    panic!("not yet implemented");
} // end of try_decomposition


#[cfg_attr(doc, katexit::katexit)]
/// computes a decomposition of graph in blocks of vertices of decreasing density. 
/// 
/// First we search maximal densest subsets $S_{i}$ are deduced from the isotonic regression.  
/// Then we get a diminishingly dense decomposition of the graph in blocks $B_{i}$.  
/// The blocks satisfy:
///  - $B_{i} \subset B_{i+1}$ 
///  - $B_{0}=\emptyset , B_{max}=V$ where $V$ is the set of vertices of G.
///  - each block must be stable.    
/// 
///  The blocks are defined by $B_{i} = B_{i-1} \cup S_{i}$
pub struct StableDecomposition {
    /// list of disjoint maximal densest subsets deduced from the isotonic regression.
    /// stable blocks filtered . Give for each point the $S_{i}$ to which the point belongs.
    /// Alternativeley s`[i`] contains the densest stable block to which node i belongs
    s : Vec<u32>,
    /// list of numblocks as given in s but sorted in increasing num
    index : Vec<usize>,
    /// for each block give its beginning position in index. 
    /// block[0] begins at 0, block[1] begins at index[block_start[1]]
    block_start : Vec<usize>,
} // end of struct StabeDecomposition


impl StableDecomposition {
    pub fn new(s : Vec<u32>) -> Self {
        log::info!("StableDecomposition::new");
        let index = s.mergesort_indexed().rank(true);
        log::debug!("sorted s : {}, {}", s[index[0]], s[index[1]]);
        let last_block = s[index[index.len()-1]] as usize;
        log::debug!("last block : {}", last_block);
        //
        let mut block_start  = Vec::<usize>::with_capacity(last_block);
        let mut current_block = 0;
        for i in 0..s.len() {
            log::debug!("i : {}, point : {}, current_block : {}", i, index[i], current_block);
            if i == 0 || s[index[i]]  > current_block {
                if s[index[i]] > 0 {
                    log::info!("bloc : {} has size : {}", current_block, i - block_start[current_block as usize]);
                }
                block_start.push(i);
                current_block = s[index[i]];
            }
        }
        //
        if log_enabled!(Level::Debug) {
            log::debug!(" indexes : {:?}", index);
            log::debug!(" block start : {:?}", block_start);
        }
        //
        StableDecomposition{s, index, block_start}
    }  // end of new StableDecomposition


    /// get densest block for a node
    pub fn get_densest_block(&self, node: usize) -> usize { self.s[node] as usize}



    /// returns the points in a given block
    pub fn get_block_points(&self, blocknum : usize) -> Result<Vec<usize>, anyhow::Error> {
        //
        log::debug!(" in get_block_points, blocnum : {}", blocknum);
        //
        if blocknum > self.block_start.len() {
            return Err(anyhow!("too large num of block"));
        }
        let nb_block = self.block_start.len();
        let mut pt_list = Vec::<usize>::new();
        let start = self.block_start[blocknum];
        let end = if blocknum < nb_block - 1 { self.index[blocknum+1]} else { self.index.len() };
        log::debug!("bloc start in index : {}, end (excluded) : {}", start, end);
        for p in start..end {
            pt_list.push(self.index[p]);
        }
        if log_enabled!(Level::Debug) {
            log::debug!(" indexes : {:?}", self.index);
            log::debug!(" block start : {:?}", start);
        }
        Ok(pt_list)
    } // end of get_block_points
} // end of impl StableDecomposition







/// computes a decomposition of graph in blocks of vertices of decreasing density. 
pub fn approximate_decomposition<'a, N, F>(graph : &'a Graph<N, F, Undirected>) 
        where  F : Float + std::fmt::Debug + std::iter::Sum + FromPrimitive 
                        + std::ops::AddAssign + std::ops::DivAssign + std::ops::SubAssign + Sync + Send ,
               N : Copy {
    //
    let cpu_start = ProcessTime::now();
    let sys_start = SystemTime::now();
    //
    let nbiter = 100;
    let alpha_r = get_alpha_r(graph, nbiter);
    log::info!("frank_wolfe (fn get_alpha_r) sys time(s) {:.2e} cpu time(s) {:.2e}", 
            sys_start.elapsed().unwrap().as_secs(), cpu_start.elapsed().as_secs());
    //
    let alpha = alpha_r.get_alpha();
    let r = alpha_r.get_r();
    //
    let mut y : Vec<F> = (0..r.len()).into_iter().map(|_| F::zero()).collect();
    for i in 0..alpha.len() {
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
    let _res = iso_regression.check_blocks();

    // we try to get blocks. Must make union of blocks to get increasing sequence of blocks
    // and check their stability
    let _numblocks = get_point_blocnum(&iso_regression);
    log::info!("isotonic regression made nb_blocks : {}", iso_regression.get_nb_block());
    //
    log::info!(" unionization and stability check");
    let cpu_start = ProcessTime::now();
    let sys_start = SystemTime::now();
    let s = check_stability(graph,&alpha_r, &iso_regression);
    //
    log::info!("frank_wolfe (fn get_alpha_r) sys time(s) {:.2e} cpu time(s) {:.2e}", 
            sys_start.elapsed().unwrap().as_secs(), cpu_start.elapsed().as_secs());

    // now we have in blockunion, an increasing family of stable blocks
} // end of approximate_decomposition

//==========================================================================================================


#[cfg(test)] 
mod tests {
   
    use super::*;

    use crate::io::csv::weighted_csv_to_graphmap;
    use crate::structure::density::pava::{PointIterator};

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    } 

    #[test]
    fn pava_miserables() {
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
        let alpha_r = get_alpha_r(&graph, 50);
        let alpha = alpha_r.get_alpha();
        let r = alpha_r.get_r();
        //
        let mut y : Vec<f64> = (0..r.len()).into_iter().map(|_| 0.).collect();
        for i in 0..alpha.len() {
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
        let points : Vec<Point<f64>> = (0..r.len()).into_iter().map(|i| Point::new(-r[i], y[i])).collect();
        let iso_regression = IsotonicRegression::new_descending(&points);
        let res_regr = iso_regression.do_isotonic();
        if res_regr.is_err() {
            log::error!("approximate_decomposition failed in iso_regression regression");
            std::process::exit(1);
        }
        let _res = iso_regression.check_blocks();
        // check iterator on bloc2
        let block = iso_regression.get_block(4).unwrap();
        log::debug!("\n block dump");
        block.dump();
        log::debug!("\n block iteration");
        let mut blockiter = PointIterator::new(&block, iso_regression.get_point_index());
        let mut nb_points_in = 0;
        while let Some(point) = blockiter.next() {
            log::debug!("point : {:?}", point);
            nb_points_in += 1;
        }
        assert_eq!(nb_points_in, block.get_nb_points());
        //
        let _pointblockloc = PointBlockLocator::new(&iso_regression);
        let _block = iso_regression.get_block(0).unwrap();
        // dump degrees of each nodes
        let nb_nodes = graph.node_count();
        log::info!("dump degrees , nb_nodes : {}", nb_nodes);
        for node in 0..nb_nodes {
            let degree = graph.neighbors(NodeIndex::new(node)).count();
            log::info!(" node : {}, degree : {}", node, degree);
        }
        let decomposition = check_stability(&graph, &alpha_r, &iso_regression);
        let blocnum = 0;
        let block = decomposition.get_block_points(blocnum).unwrap();
        log::info!("pava_miserables : points of block : {} , {:?}", blocnum, block);
    }  // end of pava_miserables




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
        let alpha_r = get_alpha_r(&graph, 50);
        let r = alpha_r.get_r();
        log::debug!("r : {:?}", &r[0..20]);

        //
        approximate_decomposition(&graph);
    }
} // end of mod tests