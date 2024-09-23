//! density decomposition algorithm.
//!
//! The graph representation relies on petgraph.
//! The function doing the work is [approximate_decomposition]. The result is returned in the structure [StableDecomposition]

use cpu_time::ProcessTime;
/// We use Graph representation from petgraph.
///
/// For Frank-Wolfe iterations, we need // access to edges.
///
use std::time::SystemTime;

use num_traits::{float::*, FromPrimitive};

// synchronization primitive
use parking_lot::RwLock;
use rayon::prelude::*;
use std::sync::Arc;

use hdrhistogram::Histogram;

use ndarray::Array2;

use petgraph::graph::{EdgeReference, Graph, NodeIndex};
use petgraph::{visit::*, Undirected};

// to get sorting with index as result
//
use super::pava::{get_point_blocnum, IsotonicRegression, Point, PointBlockLocator};
use super::stable::StableDecomposition;

/// describes weight of each node of an edge.
#[derive(Copy, Clone, Debug)]
pub(crate) struct WeightSplit(f32, f32);

impl Default for WeightSplit {
    fn default() -> Self {
        WeightSplit(0., 0.)
    }
}

/// Structure describing how weight of edges is dispatched to nodes.
#[derive(Copy, Clone)]
struct EdgeSplit<'a, F> {
    edge: EdgeReference<'a, F>,
    wsplit: WeightSplit,
}

impl<'a, F> EdgeSplit<'a, F> {
    fn new(edge: EdgeReference<'a, F>, wsplit: WeightSplit) -> Self {
        EdgeSplit { edge, wsplit }
    }
}

/// Structure describing how the weight of edges is dispatched onto tis vertices.
struct AlphaR<'a, F> {
    r: Vec<F>,
    alpha: Vec<EdgeSplit<'a, F>>,
}

impl<'a, F> AlphaR<'a, F> {
    fn new(r: Vec<F>, alpha: Vec<EdgeSplit<'a, F>>) -> Self {
        AlphaR { r, alpha }
    }

    /// get r field
    pub fn get_r(&self) -> &Vec<F> {
        &self.r
    }

    /// get alpha field
    pub fn get_alpha(&self) -> &Vec<EdgeSplit<'a, F>> {
        &self.alpha
    }
} // end of impl AlphaR

/// initialize alpha and r (as defined in paper) by Frank-Wolfe algorithm
/// returns (alpha, r) alpha is dimensioned to number of edges, r is dimensioned to number of vertex
fn get_alpha_r<'a, N, F>(graph: &'a Graph<N, F, Undirected>, nbiter: usize) -> AlphaR<'a, F>
where
    F: Float
        + FromPrimitive
        + std::ops::AddAssign<F>
        + std::ops::SubAssign<F>
        + Sync
        + Send
        + std::fmt::Debug,
{
    //
    log::info!("entering Frank-Wolfe iterations");
    let cpu_start = ProcessTime::now();
    let sys_start = SystemTime::now();
    // how many nodes and edges
    let nb_nodes = graph.node_count();
    let nb_edges = graph.edge_count();
    // we will need to update r with // loops on edges.
    // We bet on low degree versus number of edges so low contention (See hogwild)
    let mut alpha: Vec<Arc<RwLock<EdgeSplit<'a, F>>>> = Vec::with_capacity(nb_edges);
    // asynchronuous is really faster! old code is kept for memory
    let asynchronuous = true;
    //
    let edges = graph.edge_references();
    for e in edges {
        let weight = e.weight();
        let split = EdgeSplit::new(
            e,
            WeightSplit(weight.to_f32().unwrap() / 2., weight.to_f32().unwrap() / 2.),
        );
        alpha.push(Arc::new(RwLock::new(split)));
    }
    // now we initialize r to 0
    let r: Vec<Arc<RwLock<f32>>> = (0..nb_nodes)
        .map(|_| Arc::new(RwLock::<f32>::new(0.)))
        .collect();
    //
    // a function that computes r from alpha after each iteration
    //
    let r_from_alpha = |r: &Vec<Arc<RwLock<f32>>>, alpha: &Vec<Arc<RwLock<EdgeSplit<'a, F>>>>| {
        // reset r to 0
        (0..nb_nodes).into_par_iter().for_each(|i| {
            *r[i].write() = 0.;
        });
        // alpha's load transferred to r
        (0..alpha.len()).into_par_iter().for_each(|i| {
            let alpha_i = alpha[i].read();
            *r[alpha_i.edge.source().index()].write() += alpha_i.wsplit.0;
            // process target
            *r[alpha_i.edge.target().index()].write() += alpha_i.wsplit.1;
        });
    };
    //
    // We dispatch alpha to r
    r_from_alpha(&r, &alpha);
    //
    for iter in 0..nbiter {
        let gamma = 2. / (2. + iter as f32);
        //
        if iter % 100 == 0 {
            log::info!("iteration : {}, {:.3e}", iter, gamma);
        }
        //
        (0..alpha.len()).into_par_iter().for_each(|i| {
            let mut delta_i = WeightSplit::default();
            let mut alpha_i = alpha[i].write();
            let source = alpha_i.edge.source();
            let target = alpha_i.edge.target();
            let r_source = *r[source.index()].read();
            let r_target = *r[target.index()].read();
            // get edge node with min r. The smaller gets the weight
            if r_source < r_target {
                delta_i.0 = alpha_i.edge.weight().to_f32().unwrap(); // andd delta_i.1 = 0.;
                alpha_i.wsplit.0 = (1. - gamma) * alpha_i.wsplit.0 + gamma * delta_i.0;
                alpha_i.wsplit.1 = (1. - gamma) * alpha_i.wsplit.1;
                if asynchronuous {
                    *r[source.index()].write() += (delta_i.0 - alpha_i.wsplit.0) * gamma;
                    *r[target.index()].write() += (0. - alpha_i.wsplit.1) * gamma;
                }
            } else if r_target < r_source {
                delta_i.1 = alpha_i.edge.weight().to_f32().unwrap(); // and delta_i.0 = 0.
                alpha_i.wsplit.0 = (1. - gamma) * alpha_i.wsplit.0;
                alpha_i.wsplit.1 = (1. - gamma) * alpha_i.wsplit.1 + gamma * delta_i.1;
                if asynchronuous {
                    *r[source.index()].write() += (0. - alpha_i.wsplit.0) * gamma;
                    *r[target.index()].write() += (delta_i.1 - alpha_i.wsplit.1) * gamma;
                }
            }
            // else e do nothing!
        }); // end of // computation of
            // now we recompute r
        if !asynchronuous {
            r_from_alpha(&r, &alpha);
        }
    } // end of // loop on edges
      // We do not need locks any more, simplify
    let r_s: Vec<F> = r.iter().map(|v| F::from(*v.read()).unwrap()).collect();
    let alpha_s: Vec<EdgeSplit<'a, F>> = alpha.iter().map(|a| a.read().clone()).collect();
    //
    log::info!(
        "frank_wolfe (fn get_alpha_r) sys time(s) {:.2e} cpu time(s) {:.2e}",
        sys_start.elapsed().unwrap().as_secs(),
        cpu_start.elapsed().as_secs()
    );
    //
    return AlphaR::new(r_s, alpha_s);
} // end of get_alpha_r

/// returns the dgree of a node
pub fn get_degree_undirected<N, F>(
    graph: &Graph<N, F, Undirected>,
    rank: usize,
) -> Result<usize, anyhow::Error> {
    let nb_nodes = graph.node_count();
    if rank >= nb_nodes {
        return Err(anyhow::anyhow!("bad index, nb_nodes : {nb_nodes}"));
    }
    let neighbours = graph.neighbors(NodeIndex::new(rank));
    Ok(neighbours.count())
} // end of get_degree_undirected

/// check stability of a given vertex block with respect to alfar (algo 2 of Danisch paper)
fn check_stability<'a, F: Float + std::fmt::Debug, N>(
    graph: &'a Graph<N, F, Undirected>,
    alphar: &'a AlphaR<'a, F>,
    iso_regression: &'a IsotonicRegression<F>,
) -> StableDecomposition
where
    F: Float
        + std::iter::Sum
        + FromPrimitive
        + std::ops::DivAssign
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::fmt::Debug
        + Sync
        + Send,
    N: Copy,
{
    //
    let cpu_start = ProcessTime::now();
    let sys_start = SystemTime::now();
    //
    let nb_reg_blocks = iso_regression.get_nb_block();
    let nb_nodes = graph.node_count();
    let mut degrees = (0..nb_nodes).map(|_| 0).collect::<Vec<u32>>();
    let pointblocklocator = PointBlockLocator::new(iso_regression);
    //
    let mut block_transition = Array2::<f32>::zeros((nb_reg_blocks, nb_reg_blocks));
    let mut block_size = (0..nb_reg_blocks).map(|_| 0usize).collect::<Vec<usize>>();
    //
    let alfa_tmp = alphar.get_alpha().clone();
    let mut r = alphar.get_r().clone();
    let mut r_test = alphar.get_r().clone();
    // initialize stable_numblocks to sthing  impossible so we know if some points leap through a hole of the algo
    let mut stable_numblocks: Vec<u32> = (0..r.len()).map(|_| (nb_reg_blocks + 1) as u32).collect();
    let mut points_waiting = Vec::<usize>::with_capacity(r.len());
    let mut block_waiting: u32 = 0;
    //
    for numbloc in 0..nb_reg_blocks {
        log::debug!("\n stability check for block : {}", numbloc);
        let block = iso_regression.get_block(numbloc).unwrap();
        block_size[numbloc] = block.get_nb_points();
        // TODO this iteration can be made // if necessary
        let ptiter = block.get_point_iter();
        for (&_pt, rank_pt) in ptiter {
            let pt_idx = NodeIndex::new(rank_pt);
            points_waiting.push(pt_idx.index());
            // rank guve us the index in graph
            let mut degree = 0;
            let mut neighbours = graph.neighbors(pt_idx).detach();
            // is neighbor in block
            while let Some((edge_idx, neighbor)) = neighbours.next(graph) {
                degree += 1;
                let neighbor_u = neighbor.index();
                // TODO >= or ==
                let neighbour_block = pointblocklocator.get_point_block_num(neighbor_u).unwrap();
                block_transition[(numbloc, neighbour_block)] += 1.;
                if neighbour_block >= numbloc + 1 {
                    // then we get edge corresponding to (pt , neighbor), modify alfa. Cannot fail
                    let edge = graph.edge_endpoints(edge_idx).unwrap();
                    // we must check for order. We have the same order of of the 2-uple in wsplit and in edge
                    if edge.0 == pt_idx && edge.1 == neighbor {
                        r_test[edge.0.index()] -=
                            F::from(alfa_tmp[edge_idx.index()].wsplit.0).unwrap();
                        r_test[edge.1.index()] +=
                            F::from(alfa_tmp[edge_idx.index()].wsplit.0).unwrap();
                    } else if edge.0 == neighbor && edge.1 == pt_idx {
                        r_test[edge.1.index()] -=
                            F::from(alfa_tmp[edge_idx.index()].wsplit.1).unwrap();
                        r_test[edge.0.index()] +=
                            F::from(alfa_tmp[edge_idx.index()].wsplit.1).unwrap();
                    } else {
                        panic!("should not happen");
                    }
                }
            } // end while on neighbours
              // affect degree for this node
            assert_eq!(degrees[pt_idx.index()], 0); // consistency check...
            degrees[pt_idx.index()] = degree;
        } // end while on point in blocks
          // we must check that r is greater on block than outside
        let mut min_in_block = F::max_value();
        let mut max_not_in_block = F::zero();
        (0..r_test.len()).for_each(|i| {
            let b = pointblocklocator.get_point_block_num(i).unwrap();
            if b <= numbloc {
                min_in_block = min_in_block.min(r_test[i]);
            } else if b > numbloc {
                // we
                max_not_in_block = max_not_in_block.max(r_test[i]);
            }
        });
        log::trace!(
            "stability result  bloc : {}, min in block {:?}, max out block {:?}",
            numbloc,
            min_in_block,
            max_not_in_block
        );
        if min_in_block > max_not_in_block {
            // got a stable block, we reset r with r_test
            log::info!(
                "stable bloc : {}, regr block : {:?}, min in block {:?}, max out block {:?}",
                block_waiting,
                numbloc,
                min_in_block,
                max_not_in_block
            );
            (0..r_test.len()).for_each(|i| r[i] = r_test[i]);
            for p in &points_waiting {
                stable_numblocks[*p] = block_waiting;
            }
            block_waiting += 1;
            points_waiting.clear();
            if numbloc >= nb_reg_blocks - 1 {
                log::debug!("check stability examined all initial regreesion blocks");
                break;
            }
        } else {
            // reset r_test to last stable state
            (0..r_test.len()).for_each(|i| r_test[i] = r[i]);
        }
        // if we are in the last regression_blocks we treat points_waiting
        if !points_waiting.is_empty() && numbloc == nb_reg_blocks - 1 {
            log::debug!("treating last block with waiting_points");
            for p in &points_waiting {
                stable_numblocks[*p] = block_waiting;
            }
        }
    } // end of loop on initial_blocks
      //
    log::info!(
        "\n check stability sys time(s) {:.2e} cpu time(s) {:.2e}",
        sys_start.elapsed().unwrap().as_secs(),
        cpu_start.elapsed().as_secs()
    );
    // a check
    assert_eq!(points_waiting.len(), 0);
    for i in 0..r.len() {
        if stable_numblocks[i] >= (nb_reg_blocks + 1) as u32 {
            log::error!(
                " point is not affected a good block, point : {}, stable block : {}",
                i,
                stable_numblocks[i]
            );
            iso_regression.check_blocks();
            std::panic!();
        }
    }
    // dump stable_numblocks
    if log::log_enabled!(log::Level::Debug) {
        log::debug!("dumping stable_numblocks");
        for (p, block) in stable_numblocks.iter().enumerate() {
            log::debug!("point : {},  bloc : {}", p, block);
        }
    }
    //
    // process matrix of block_transition
    // for each block i get fraction of edge out. i.e going to j. We get a transition probability for each block
    //
    let mut fraction_out = (0..nb_reg_blocks).map(|_| 0f32).collect::<Vec<f32>>();
    let mean_block_size = block_size.iter().sum::<usize>() as f32 / nb_reg_blocks as f32;
    for i in 0..nb_reg_blocks {
        let block_degree = block_transition.row(i).iter().sum::<f32>();
        fraction_out[i] = (i + 1..nb_reg_blocks)
            .fold(0., |acc: f32, j| acc + block_transition[(i, j)])
            / block_degree;
        log::info!(" block {i}, fraction out : {:.3e}", fraction_out[i]);
        block_transition
            .row_mut(i)
            .iter_mut()
            .zip(0usize..)
            .for_each(|v| *v.0 /= block_degree);
    }
    log::info!(" mean block size : {:?}", mean_block_size);
    log::info!("\n block_transition : {:?}", &block_transition);

    // now we can return stable_numblocks
    StableDecomposition::new(stable_numblocks, degrees, block_transition)
} // end check_stability

/// computes an approximate decomposition of graph in blocks of vertices of decreasing density.  
/// nb_iter is the number of iteration asked for. A standard value is 500.
pub fn approximate_decomposition<N, F>(
    graph: &Graph<N, F, Undirected>,
    nbiter: usize,
) -> StableDecomposition
where
    F: Float
        + std::fmt::Debug
        + std::iter::Sum
        + FromPrimitive
        + std::ops::AddAssign
        + std::ops::DivAssign
        + std::ops::SubAssign
        + Sync
        + Send,
    N: Copy,
{
    //
    let cpu_start = ProcessTime::now();
    let sys_start = SystemTime::now();
    //
    let alpha_r = get_alpha_r(graph, nbiter);
    log::info!(
        "fn get_alpha_r sys time(s) {:.2e} cpu time(s) {:.2e}",
        sys_start.elapsed().unwrap().as_secs(),
        cpu_start.elapsed().as_secs()
    );
    //
    let alpha = alpha_r.get_alpha();
    let r = alpha_r.get_r();
    //
    let mut y: Vec<F> = (0..r.len()).map(|_| F::zero()).collect();
    for esplit in alpha {
        let node_max = if esplit.wsplit.0 > esplit.wsplit.1 {
            esplit.edge.source().index()
        } else {
            esplit.edge.target().index()
        };
        y[node_max] += *esplit.edge.weight();
    } // end of for i
      // go to PAVA algorithm , the decomposition of y in blocks makes a tentative decomposition
      // as -r increases , y decreases. We begin algo by densest blocks!
    let points: Vec<Point<F>> = (0..r.len()).map(|i| Point::new(-r[i], y[i])).collect();
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
    log::info!(
        "isotonic regression made nb_blocks : {}",
        iso_regression.get_nb_block()
    );
    //
    log::info!(" unionization and stability check");
    let cpu_start = ProcessTime::now();
    let sys_start = SystemTime::now();
    let s = check_stability(graph, &alpha_r, &iso_regression);
    //
    log::info!(
        "\n approximate_decomposition sys time(s) {:.2e} cpu time(s) {:.2e}",
        sys_start.elapsed().unwrap().as_secs(),
        cpu_start.elapsed().as_secs()
    );
    //
    s
} // end of approximate_decomposition

/// log::info histograms of degrees of incremental blocks S_i whose union make B_i
#[allow(unused)]
pub fn get_degree_statistics<N, F>(graph: &Graph<N, F, Undirected>, stable: &StableDecomposition) {
    //
    let quantiles = vec![0.05, 0.25, 0.5, 0.75, 0.95];
    log::info!("quantiles used : {:?}", quantiles);
    let nb_blocks = stable.get_nb_blocks();
    for i in 0..nb_blocks {
        get_block_degree_statistics(graph, stable, &quantiles, i).unwrap();
    }
} // end of get_degree_statistics

/// log::info hsitograms of degree in block of StableDecomposition
pub fn get_block_degree_statistics<N, F>(
    graph: &Graph<N, F, Undirected>,
    stable: &StableDecomposition,
    quantiles: &[f64],
    blocknum: usize,
) -> Result<(), ()> {
    //
    let nb_blocks = stable.get_nb_blocks();
    if blocknum >= nb_blocks {
        return Err(());
    }
    let block = stable.get_block_points(blocknum).unwrap();
    let mut histo = Histogram::<u64>::new(2).unwrap();
    for p in &block {
        histo += get_degree_undirected(graph, *p).unwrap() as u64;
    }
    let degrees = quantiles
        .iter()
        .map(|f| histo.value_at_quantile(*f))
        .collect::<Vec<u64>>();
    log::info!(" block degrees: {blocknum}, degrees : {:?} ", degrees);
    //
    Ok(())
} // end of get_block_degree_statistics

//==========================================================================================================

#[cfg(test)]
mod tests {

    use super::*;

    use crate::io::csv::weighted_csv_to_graphmap;
    use crate::structure::density::pava::PointIterator;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn pava_miserables() {
        log_init_test();
        //
        log::debug!("in algodens density_miserables");
        let path = std::path::Path::new(crate::DATADIR)
            .join("moreno_lesmis")
            .join("out.moreno_lesmis_lesmis");
        log::info!("\n\n algodens::density_miserables, loading file {:?}", path);
        let res = weighted_csv_to_graphmap::<u32, f64, Undirected>(&path, b' ');
        if res.is_err() {
            log::error!("algodens::density_miserables failed in csv_to_trimat");
            assert_eq!(1, 0);
        }
        // now we can convert into a Graph
        let graph = res.unwrap().into_graph();
        // check get_alpha_r
        let alpha_r = get_alpha_r(&graph, 400);
        let alpha = alpha_r.get_alpha();
        let r = alpha_r.get_r();
        //
        let mut y: Vec<f64> = (0..r.len()).into_iter().map(|_| 0.).collect();
        for i in 0..alpha.len() {
            let node_max = if alpha[i].wsplit.0 > alpha[i].wsplit.1 {
                alpha[i].edge.source().index()
            } else {
                alpha[i].edge.target().index()
            };
            y[node_max] += *alpha[i].edge.weight();
        } // end of for i
          // go to PAVA algorithm , the decomposition of y in blocks makes a tentative decomposition
          // as -r increases , y decreases. We begin algo by densest blocks!
        let points: Vec<Point<f64>> = (0..r.len())
            .into_iter()
            .map(|i| Point::new(-r[i], y[i]))
            .collect();
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
            log::info!(" node : {}, degree : {} r : {}", node, degree, r[node]);
        }
        let decomposition = check_stability(&graph, &alpha_r, &iso_regression);
        let nb_blocks = decomposition.get_nb_blocks();
        log::info!("pava_miserables got nb_block : {nb_blocks}");
        for blocnum in 0..nb_blocks {
            let block = decomposition.get_block_points(blocnum).unwrap();
            log::info!(
                "pava_miserables : points of block : {} , {:?}",
                blocnum,
                block
            );
        }
        //
        get_degree_statistics(&graph, &decomposition);
    } // end of pava_miserables

    #[test]
    fn density_miserables() {
        log_init_test();
        //
        log::debug!("in algodens density_miserables");
        let path = std::path::Path::new(crate::DATADIR)
            .join("moreno_lesmis")
            .join("out.moreno_lesmis_lesmis");
        log::info!("\n\n algodens::density_miserables, loading file {:?}", path);
        let res = weighted_csv_to_graphmap::<u32, f64, Undirected>(&path, b' ');
        if res.is_err() {
            log::error!("algodens::density_miserables failed in csv_to_trimat");
            assert_eq!(1, 0);
        }
        // now we can convert into a Graph
        let graph = res.unwrap().into_graph();
        //
        let nb_iter = 100;
        let decomposition = approximate_decomposition(&graph, nb_iter);
        let nb_blocks = decomposition.get_nb_blocks();
        log::info!("pava_miserables got nb_block : {nb_blocks}");
        // dump degrees of each nodes
        let nb_nodes = graph.node_count();
        log::info!("dump degrees , nb_nodes : {}", nb_nodes);
        for node in 0..nb_nodes {
            let degree = graph.neighbors(NodeIndex::new(node)).count();
            log::info!(" node : {}, degree : {}", node, degree);
        }
        // get blocksizes
        let mut blocksize = Vec::<usize>::new();
        for blocnum in 0..nb_blocks {
            let bsize = decomposition.get_nbpoints_in_block(blocnum).unwrap();
            blocksize.push(bsize);
            log::info!("density_miserables : points of block : {blocnum} , {bsize}");
        }
        for blocnum in 0..nb_blocks {
            let block = decomposition.get_block_points(blocnum).unwrap();
            assert_eq!(block.len(), blocksize[blocnum]);
            log::info!(
                "pava_miserables : points of block : {} , {:?}",
                blocnum,
                block
            );
        }
    }
} // end of mod tests
