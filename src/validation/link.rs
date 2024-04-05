//! a simple link prediction to test Embeddeding
//!
//! We implement precision measure as described in :  
//!       - Link Prediction in complex Networks : A survey.  
//!            Lü, Zhou. Physica 2011.  
//!
//! The scoring function for similarity is based upon the distance relative to the Embedded being tested
//! Jaccard for nodesketch and L2 for Atp
//!
//! Deletion of edges is done differently depending on the symetry/asymetry of the graph.  
//! For a symetric graph edge i->j and j->i are deleted/kept together and for an asymetric graph they are treated independantly.
//! It is possible to treat edge deletion for a symetric graph as in the asymetric. See [crate::embed]
//!
//!

/// We first implement precision measure as described in
///       - Link Prediction in complex Networks : A survey
///             Lü, Zhou. Physica 2011
///
///  Another reference is :
///       - Link prediction problem for social networks
///             David Liben-Nowell, J Kleinberg 2007
use log::log_enabled;

use cpu_time::ProcessTime;
use std::time::SystemTime;

use rand::distributions::{Distribution, Uniform};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use indexmap::IndexSet;
use std::collections::{HashMap, HashSet};

use sprs::{CsMatI, TriMatI};

use hdrhistogram::Histogram;
use quantiles::ckms::CKMS;
use rayon::prelude::*;

use crate::embed::tools::{correlation::*, degrees::*, edge::Edge, edge::IN, edge::OUT};
use crate::embedding::EmbeddedT;

pub enum ValidationMode {
    NODELABEL,
}

// ======================================================================================

// filter out edge with proba delete_proba
// The return a tuple containing the matrix of the graph with some edge deleted and a perfect hash on deleted edges
fn filter_csmat<F>(
    csrmat: &CsMatI<F, usize>,
    delete_proba: f64,
    symetric: bool,
    rng: &mut Xoshiro256PlusPlus,
) -> (TriMatI<F, usize>, IndexSet<(usize, usize)>)
where
    F: Default + Copy,
{
    //
    log::debug!(
        "filter_csmat symetric mode : {}, delete proba : {:.3e}",
        symetric,
        delete_proba
    );
    //
    assert!(delete_proba < 1. && delete_proba > 0.);
    log::debug!(
        "filter_csmat, delete proba {}, symetric : {}",
        delete_proba,
        symetric
    );
    // get total in + out degree
    let nb_nodes = csrmat.rows();
    let nb_edge = csrmat.nnz();
    let mut deleted_edge =
        IndexSet::with_capacity((nb_edge as f64 * delete_proba).round() as usize);

    //
    let uniform = Uniform::<f64>::new(0., 1.);
    let mut rows = Vec::<usize>::with_capacity(nb_nodes);
    let mut cols = Vec::<usize>::with_capacity(nb_nodes);
    let mut values = Vec::<F>::with_capacity(nb_nodes);
    // we need degrees as we cannot delete edges if they are the last
    // as we cannot train anything if nodes is disconnected
    let mut degrees = get_csmat_degrees::<F>(csrmat);
    //
    log::info!(
        "mean in degree : {:.3e}",
        degrees.iter().map(|degree| degree.d_in).sum::<u32>() as f64 / nb_nodes as f64
    );
    //
    let mut discarded: usize = 0;
    let mut nb_isolation_not_discarded: usize = 0;
    let nb_to_discard = (nb_edge as f64 * delete_proba) as usize;
    //
    log::debug!(
        "csrmat nb edge : {}, number to discard {}",
        nb_edge,
        nb_to_discard
    );

    let mut csmat_iter = csrmat.iter();
    while let Some((value, (row, col))) = csmat_iter.next() {
        let xsi = uniform.sample(rng);
        let mut discard = false;
        if xsi < delete_proba {
            // We avoid making isolated node
            if symetric {
                // we check only for row < col as we will force coherent deletion at end of scan
                if degrees[row].d_out > 1 && degrees[col].d_in > 1 && row < col {
                    discard = true;
                } else if row < col {
                    // count not discarding just for degree reason (and not for above diag)
                    nb_isolation_not_discarded += 1;
                }
            } else {
                if (degrees[row].degree_out() > 1 || degrees[row].degree_in() > 0)
                    && (degrees[col].degree_in() > 1 || degrees[col].degree_out() > 0)
                    && row != col
                {
                    discard = true;
                } else {
                    nb_isolation_not_discarded += 1;
                }
            }
        }
        if !discard {
            // we keep edge in triplets
            if !symetric {
                rows.push(row);
                cols.push(col);
                values.push(*value);
            }
            // in symetric mode !discard is significant only when row < col!
            else {
                if row < col {
                    rows.push(row);
                    cols.push(col);
                    values.push(*value);
                    rows.push(col);
                    cols.push(row);
                    values.push(*value);
                } else if row == col {
                    rows.push(row);
                    cols.push(col);
                    values.push(*value);
                }
            }
        } else {
            assert!(row != col);
            log::trace!("link:validation deleting edge {}->{}", row, col);
            deleted_edge.insert((row, col));
            degrees[row].d_out -= 1;
            degrees[col].d_in -= 1;
            discarded += 1;
            if symetric {
                // maintian coherence of in and out degree in symetric case, and delete symetric edge!
                degrees[row].d_in -= 1;
                degrees[col].d_out -= 1;
                deleted_edge.insert((col, row));
                discarded += 1;
            }
            if log_enabled!(log::Level::Debug) || log_enabled!(log::Level::Trace) {
                if !(degrees[row].d_in > 0 || degrees[row].d_out > 0) {
                    log::error!("degrees node row : {} degree = {:?}", row, degrees[row]);
                }
                if !(degrees[col].d_in > 0 || degrees[col].d_out > 0) {
                    log::error!("degrees node col : {} degree = {:?}", row, degrees[row]);
                }
            }
            // chech we do not have isolated nodes.
            assert!(degrees[row].d_in > 0 || degrees[row].d_out > 0);
            assert!(degrees[col].d_in > 0 || degrees[col].d_out > 0);
        }
    } // end while
      //
    assert_eq!(rows.len(), cols.len());
    assert_eq!(cols.len(), values.len());
    //
    log::info!(
        " ratio discarded = {:.3e} nb edges after deletion : {}",
        discarded as f64 / (nb_edge as f64),
        values.len()
    );
    if nb_isolation_not_discarded > 0 {
        log::info!(
            " ratio not discarded to avoid disconnected node = {:.3e}",
            nb_isolation_not_discarded as f64 / (nb_edge as f64)
        );
    }
    //
    let trimat = TriMatI::<F, usize>::from_triplets((nb_nodes, nb_nodes), rows, cols, values);
    //
    (trimat, deleted_edge)
} // end of filter_csmat

/// filters the full graph matrix edges with with rate delete_proba.
/// embed the filtered and sort all (seen and not seen) edges
/// according to Embedded similarity function and return.
/// precision : ratio of really deleted out / over the best 100 unused for the training (i.e precision of the iteration)
/// recall : ratio if
/// type G is necessary beccause we embed in possibly different type than F. (for example in Array<usize> with nodesketch)
///
/// return (precision, recall)
fn one_precision_iteration<F, G, E>(
    csmat: &CsMatI<F, usize>,
    delete_proba: f64,
    symetric: bool,
    embedder: &dyn Fn(TriMatI<F, usize>) -> E,
    rng: &mut Xoshiro256PlusPlus,
) -> (f64, f64)
where
    F: Default + Copy + std::marker::Sync,
    E: EmbeddedT<G> + std::marker::Sync,
{
    //
    // filter
    let (trimat, deleted_edges) = filter_csmat(csmat, delete_proba, symetric, rng);
    log::debug!(
        "\n\n one_precision_iteration : nb_deleted edges : {}",
        deleted_edges.len()
    );
    // We construct the list of edges not in reduced graph
    let nb_nodes = csmat.shape().0;
    let max_degree = csmat.degrees().into_iter().max().unwrap();
    let mut embedded_edges = Vec::<Edge>::with_capacity(nb_nodes * nb_nodes);
    // filter out self loops and edges in trimat so we keep all edges except the train set
    let f_i = |i: usize| -> Vec<Edge> {
        let mut edges_i = Vec::<Edge>::with_capacity(max_degree);
        for j in 0..nb_nodes {
            if j != i && (&trimat).find_locations(i, j).len() == 0usize {
                edges_i.push(Edge {
                    0: i,
                    1: j,
                    2: 0f64,
                });
            }
        }
        edges_i
    };
    let mut row_embedded: Vec<Vec<Edge>> = (0..nb_nodes).into_par_iter().map(|i| f_i(i)).collect();
    for i in 0..nb_nodes {
        log::trace!("row i : {}, row_embedded {:?}", i, row_embedded[i]);
        embedded_edges.append(&mut row_embedded[i]);
    }
    //
    // embed (to be passed as a closure)
    //
    let embedded = &embedder(trimat);
    // now we can compute all edge valuation (similarities betwen nodes pairs) and sort in increasing order
    // find how many deleted edges are in upper part of the sorted edges.
    for edge in &mut embedded_edges {
        edge.2 = embedded.get_noderank_distance(edge.0, edge.1);
    }
    //
    embedded_edges.sort_unstable_by(|ea, eb| ea.2.partial_cmp(&eb.2).unwrap());
    log::debug!(
        "one_precision_iteration: complete graph minus deleted edges in  : {}",
        embedded_edges.len()
    );
    log::debug!(
        "nb embedded edges : {}, smallest : {:?}, largest : {:?}",
        embedded_edges.len(),
        embedded_edges[0],
        embedded_edges[embedded_edges.len() - 1]
    );
    //
    //  TODO parametrize? we give precision at 95% of deleted
    let retrieved: usize = (0.95 * deleted_edges.len() as f64) as usize;
    embedded_edges.truncate(retrieved);
    log::debug!(
        "\n keeping {}, smallest non diagonal edges remaining {:?} ... {:?}",
        retrieved,
        embedded_edges[0],
        embedded_edges[embedded_edges.len() - 1]
    );
    // how many deleted edges (and so not learned in the embedding) are in the edges list with smallest distance
    let mut h_edges = HashMap::<(usize, usize), f64>::with_capacity(embedded_edges.len());
    for edge in embedded_edges {
        h_edges.insert((edge.0, edge.1), edge.2);
    }
    let mut nb_in = 0;
    for edge in &deleted_edges {
        if h_edges.contains_key(&(edge.0, edge.1)) {
            nb_in += 1;
        }
        if symetric && h_edges.contains_key(&(edge.1, edge.0)) {
            nb_in += 1;
        }
    }
    //
    let recall = nb_in as f64 / deleted_edges.len() as f64;
    let precision = nb_in as f64 / retrieved as f64;
    assert!(precision <= 1.);
    let f1: f64 = 2. * recall * precision / (precision + recall);
    log::debug!("one_precision_iteration : precison {:.3e}, recall : {:.3e} nb_deleted : {}, nb_retrieved : {}", precision, recall, deleted_edges.len(), retrieved);
    log::info!("f1 = {:.3e}", f1);
    //
    (precision, recall)
} // end of one_precision_iteration

//==

/// estimate precision on nbiter iterations and edge deletion probability.
///
/// Precision estimation is a costly operation as it requires computing the whole (nbnode, nbnode) matrix similarity
/// between embedded nodes and sorting them to compare with the deleted edges. So precision uses a comparison of each deleted edge
/// with most probables inexistant edges.  
///
/// AUC instead samples, for each deleted edge, a random inexistant edge from the original graph and compute its probability in the embedded graph
/// count number of times the deleted edge is more probable than the deleted.
///
pub fn estimate_precision<F, G, E>(
    csmat: &CsMatI<F, usize>,
    nbiter: usize,
    delete_proba: f64,
    symetric: bool,
    embedder: &dyn Fn(TriMatI<F, usize>) -> E,
) -> (Vec<f64>, Vec<f64>)
where
    F: Default + Copy + Sync,
    E: EmbeddedT<G> + std::marker::Sync,
{
    //
    log::debug!(
        "in estimate_precision delete_proba : {:.3e},  symetric : {}",
        delete_proba,
        symetric
    );
    // we prepair a family of random generator for possible parallel iterations
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1235437);
    let mut rngs = Vec::<Xoshiro256PlusPlus>::with_capacity(nbiter);
    for _ in 0..nbiter {
        let new_rng = rng.clone();
        rngs.push(new_rng);
        // now we reinitialize rng by cloning it and shifting it with a large jump
        rng.jump();
    }
    //
    let mut precision = Vec::<f64>::with_capacity(nbiter);
    let mut recall = Vec::<f64>::with_capacity(nbiter);
    // TODO can be made //
    for i in 0..nbiter {
        let (iter_prec, iter_recall) =
            one_precision_iteration(csmat, delete_proba, symetric, embedder, &mut rngs[i]);
        precision.push(iter_prec);
        recall.push(iter_recall);
    }
    //
    let mean_recall: f64 = recall.iter().sum::<f64>() / recall.len() as f64;
    let mean_precision: f64 = precision.iter().sum::<f64>() / recall.len() as f64;
    log::info!(
        "\n estimate_precision   mean recall : {:.3e}, mean_precision : {:.3e}",
        mean_recall,
        mean_precision
    );
    //
    (precision, recall)
} // end of estimate_precision

/// type G is necessary beccause we embed in possibly different type than F. (for example in Array<usize> with nodesketch)
fn one_auc_iteration<F, G, E>(
    csmat: &CsMatI<F, usize>,
    delete_proba: f64,
    symetric: bool,
    embedder: &dyn Fn(TriMatI<F, usize>) -> E,
    mut rng: Xoshiro256PlusPlus,
) -> f64
where
    F: Default + Copy + std::marker::Sync,
    G: std::fmt::Debug,
    E: EmbeddedT<G> + std::marker::Sync,
{
    //
    let nb_sample = 10000;
    let mut nb_dist_equality: usize = 0;
    log::debug!(
        "\n\n in one_auc_iteration nb_sample : {:?}, delete_proba : {:.3e}, symetric {:?}",
        nb_sample,
        delete_proba,
        symetric
    );
    //
    // sample nb_sample 2-uples of edges one from the deleted edge and one inexistent (not in csmat)
    // count the number of times the distance of the first is less than the second.
    // filter
    let (trimat, deleted_edges) = filter_csmat(csmat, delete_proba, symetric, &mut rng);
    // need to store trimat index before move to embedding
    let mut trimat_set = HashSet::<(usize, usize)>::with_capacity(trimat.nnz());
    for triplet in trimat.triplet_iter() {
        trimat_set.insert((triplet.1 .0, triplet.1 .1));
    }
    //
    // embedder (passed as a closure)
    //
    let embedded = &embedder(trimat);
    let mut good = 0.;
    //
    let nb_deleted = deleted_edges.len();
    let nb_nodes = csmat.shape().0;
    // as we can have large graph , mostly sparse to sample an inexistent edge we sample until we are outside csmat edges
    log::trace!("nb deleted edges : {:?}", nb_deleted);
    let del_uniform = Uniform::<usize>::from(0..nb_deleted);
    let node_random = Uniform::<usize>::from(0..nb_nodes);
    for _k in 0..nb_sample {
        let del_edge = deleted_edges
            .get_index(del_uniform.sample(&mut rng))
            .unwrap();
        let no_edge = loop {
            let i = node_random.sample(&mut rng);
            let j = node_random.sample(&mut rng);
            if i != j
                && !trimat_set.contains(&(i, j))
                && deleted_edges.get_index_of(&(i, j)).is_none()
            {
                // edge (i,j) not on diagonal and neither in trimat set neither in deleted_edges, so inexistent edge
                break (i, j);
            }
            if !symetric
                && i != j
                && !trimat_set.contains(&(j, i))
                && deleted_edges.get_index_of(&(j, i)).is_none()
            {
                // edge (i,j) not on diagonal and neither in trimat set neither in deleted_edges, so inexistent edge
                break (j, i);
            }
        };
        let dist_del_edge = embedded.get_noderank_distance(del_edge.0, del_edge.1);
        let dist_no_edge = embedded.get_noderank_distance(no_edge.0, no_edge.1);
        // debug stuff
        if log_enabled!(log::Level::Trace) {
            log::debug!(
                "distance between deleted edge nodes {} and {} : {:.3e}",
                del_edge.0,
                del_edge.1,
                dist_del_edge
            );
            log::debug!(
                "distance between no edge nodes {} and {} : {:.3e}",
                no_edge.0,
                no_edge.1,
                dist_no_edge
            );
            //            log::trace!(" dump node del_edge.0, {:?} : {:?}", no_edge.0, embedded.get_embedded_node(del_edge.0, 0));
            //            log::trace!(" dump node del_edge.1, {:?} : {:?}", no_edge.1, embedded.get_embedded_node(del_edge.1, 1));
            if dist_no_edge < dist_del_edge {
                log::debug!(
                    " node rank out del_edge.0, {:?} : {:?}",
                    del_edge.0,
                    embedded.get_embedded_node(del_edge.0, OUT)
                );
                log::debug!(
                    " node rank out del_edge.1, {:?} : {:?}",
                    del_edge.1,
                    embedded.get_embedded_node(del_edge.1, OUT)
                );
                log::debug!(
                    " node rank out no_edge.0, {:?} : {:?}",
                    no_edge.0,
                    embedded.get_embedded_node(no_edge.0, OUT)
                );
                log::debug!(
                    " node rank out no_edge.1, {:?} : {:?}",
                    no_edge.1,
                    embedded.get_embedded_node(no_edge.1, OUT)
                );
                if !symetric {
                    log::debug!(
                        " node rank in del_edge.0, {:?} : {:?}",
                        del_edge.0,
                        embedded.get_embedded_node(del_edge.0, IN)
                    );
                    log::debug!(
                        " node rank in del_edge.1, {:?} : {:?}",
                        del_edge.1,
                        embedded.get_embedded_node(del_edge.1, IN)
                    );
                    log::debug!(
                        " node rank in no_edge.0, {:?} : {:?}",
                        no_edge.0,
                        embedded.get_embedded_node(no_edge.0, IN)
                    );
                    log::debug!(
                        " node rank in no_edge.1, {:?} : {:?}",
                        no_edge.1,
                        embedded.get_embedded_node(no_edge.1, IN)
                    );
                }
            }
        }
        // end debug stuff
        if dist_del_edge < dist_no_edge {
            good += 1.;
        } else if dist_del_edge <= dist_no_edge {
            good += 0.5;
            nb_dist_equality += 1;
        }
    }
    let auc = good / nb_sample as f64;
    log::info!(" auc = {:3.e} nb dist equality : {}", auc, nb_dist_equality);
    //
    return auc;
} // end of one_auc_iteration

//
//
//

/// estimate AUC as described in Link Prediction in complex Networks : A survey
///             Lü, Zhou. Physica 2011
///
/// type G is necessary beccause we embed in a possibly different type than F. (for example in Array\<usize\> with nodesketch)
pub fn estimate_auc<F, G, E>(
    csmat: &CsMatI<F, usize>,
    nbiter: usize,
    delete_proba: f64,
    symetric: bool,
    embedder: &(dyn Fn(TriMatI<F, usize>) -> E + Sync),
) -> Vec<f64>
where
    F: Default + Copy + std::marker::Sync,
    G: std::fmt::Debug,
    E: EmbeddedT<G> + std::marker::Sync,
{
    //
    log::info!("=======================================");
    log::info!("in estimate_auc, symetric mode : {:?}", symetric);
    log::info!("=======================================");
    // we allocate as many random generator we need for each iteration for future // of iterations
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(456231);
    let mut rngs = Vec::<Xoshiro256PlusPlus>::with_capacity(nbiter);
    for _ in 0..nbiter {
        let new_rng = rng.clone();
        rngs.push(new_rng);
        // now we reinitialize rng by cloning it and shifting it with a large jump
        rng.jump();
    }
    //
    let mut auc = Vec::<f64>::with_capacity(nbiter);
    // switch in case of debugging
    let parallel = false;
    if parallel {
        auc = (0..nbiter)
            .into_par_iter()
            .map(|i| one_auc_iteration(csmat, delete_proba, symetric, embedder, rngs[i].clone()))
            .collect();
    } else {
        for i in 0..nbiter {
            let iter_auc =
                one_auc_iteration(csmat, delete_proba, symetric, embedder, rngs[i].clone());
            auc.push(iter_auc);
        }
    }
    let mean_auc: f64 = auc.iter().sum::<f64>() / (auc.len() as f64);
    let sigma2 = auc.iter().fold(0.0f64, |var: f64, x| {
        var + (*x - mean_auc) * (*x - mean_auc)
    }) / (auc.len() as f64);
    log::info!(
        "estimate_auc : mean auc : {:.3e}, std dev : {:.3e}",
        mean_auc,
        (sigma2 / (auc.len() as f64)).sqrt()
    );
    log::debug!("exiting estimate_auc");
    //
    auc
} // end of estimate_auc

//
//
/// This function is inspired by the paper:  
/// Link prediction using low-dimensional node embeddings:The measurement problem (2024)
/// See [vcmpr](https://www.pnas.org/doi/10.1073/pnas.2312527121)
///
/// It implements the paper's version and a variation, the motivation of which is to avoid null contribution of nodes
/// when there is no edge deleted incident to this node.
///
/// See also [estimate_centric_auc()] which implements a centric Auc which is more in the scale of global Auc but can also
/// reveal some risks related to be overconfident with Global Auc.
/// A discussion on link prediction can be found at TODO:
#[doc(hidden)]
pub fn estimate_vcmpr<F, G, E>(
    csmat: &CsMatI<F, usize>,
    _nbiter: usize,
    nb_edges_check: usize,
    delete_proba: f64,
    symetric: bool,
    embedder: &(dyn Fn(TriMatI<F, usize>) -> E + Sync),
) where
    F: Default + Copy + std::marker::Sync,
    G: std::fmt::Debug,
    E: EmbeddedT<G> + std::marker::Sync,
{
    log::info!("\n in estimate_vcmpr, symetric mode : {:?}", symetric);
    log::info!("===================");
    //
    let cpu_start = ProcessTime::now();
    let sys_start = SystemTime::now();
    //
    // we begin by a degree estimation and vcmpr upper bound estimation
    let degrees = get_csmat_degrees(csmat);
    let mut degree_histogram = Histogram::<u64>::new(3).unwrap();
    for d in &degrees {
        degree_histogram.record(d.d_in.into()).unwrap();
    }
    let nbslot = 40;
    let mut qs = Vec::<f64>::with_capacity(nbslot);
    for i in 1..nbslot {
        let q = i as f64 / nbslot as f64;
        qs.push(q);
    }
    qs.push(0.99);
    qs.push(0.999);
    println!("\n\n quantiles degrees of graph \n");
    println!("fraction,  degree");
    for q in qs {
        println!("{:.3e}    {}", q, degree_histogram.value_at_quantile(q));
    }
    //
    let histogram_paper = std::sync::Arc::new(std::sync::RwLock::new(CKMS::<f64>::new(0.001)));
    let histogram = std::sync::Arc::new(std::sync::RwLock::new(CKMS::<f64>::new(0.001)));
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(456231);
    // split edges into kept and discarded
    let (trimat, deleted_edges) = filter_csmat(csmat, delete_proba, symetric, &mut rng);
    // need to store trimat index before move to embedding
    let mut trimat_set = HashSet::<(usize, usize)>::with_capacity(trimat.nnz());
    for triplet in trimat.triplet_iter() {
        trimat_set.insert((triplet.1 .0, triplet.1 .1));
    }
    log::debug!(
        "trimat_set size : {} nb deleted edges : {}",
        trimat_set.len(),
        deleted_edges.len()
    );
    //
    // embedder (passed as a closure)
    //
    let embedded = &embedder(trimat);
    if !embedded.is_symetric() {
        log::warn!("method estimate_vcmpr only possible for symetric embeddings");
    }
    // sample nodes according to paper algo or mine
    let nb_nodes = csmat.shape().0;
    let nb_to_sample = 2000;
    // do we choose paper algorithm or my modification.
    let uniform_as_paper: bool = true;
    log::info!("=======================================");
    log::info!("sampling uniform : {:?}", uniform_as_paper,);
    log::info!("=======================================\n");
    //
    //=====================================================
    let selected_nodes = if uniform_as_paper {
        sample_nodes_uniform(csmat, nb_to_sample)
    } else {
        sample_nodes_by_degrees(csmat, nb_to_sample)
    };
    //
    let degrees = csmat.degrees();
    let mut mean_degree: f64 = 0.;
    let mut max_degree: usize = 0;
    for d in &degrees {
        mean_degree += (*d) as f64;
        max_degree = (*d).max(max_degree);
    }
    mean_degree /= degrees.len() as f64;
    log::info!(
        "mean degree : {:.3e}, max_degree : {}",
        mean_degree,
        max_degree
    );
    // select nodes we will test
    let nb_sampled = selected_nodes.len();
    log::info!("estimate_vcmpr nb nodes sampled : {}", nb_sampled);
    // compute precision for node i first is paper' precision , second is ours
    let f = |i: usize| -> (f64, f64) {
        // how good is the embedding of node i
        // sort edges by decreasing length
        let mut neighbours_i: Vec<(usize, f64)> = (0..nb_nodes)
            .into_par_iter()
            .map(|j| (j, embedded.get_noderank_distance(i, j)))
            .collect();
        // get neighbours sorted in increasing distance (small distance are most probable edges!)
        neighbours_i.sort_unstable_by(|a, b| (a.1).partial_cmp(&b.1).unwrap());
        //
        assert!(
            neighbours_i.first().as_ref().unwrap().1 <= neighbours_i.last().as_ref().unwrap().1
        );
        //
        if i <= 50 {
            for k in 0..20 {
                log::debug!(
                    "neighbours_i beginning, i : {} j : {}, d = {:.3e}",
                    i,
                    neighbours_i[k].0,
                    neighbours_i[k].1
                );
            }
        }
        // we have to check nb_check among deleted edges from this node
        let mut nb_found = 0;
        let mut nb_deleted = 0;
        let mut k = 0;
        for j in 1..neighbours_i.len() {
            // we bypass existing edges
            if trimat_set.contains(&(i, neighbours_i[j].0)) {
                continue;
            }
            // we know we have a potential edge
            k = k + 1;
            if deleted_edges.contains(&(i, neighbours_i[j].0)) {
                log::debug!(
                    " node {}, degree : {}, edge rank : {}, neighbour : {},, dist : {:.3e}",
                    i,
                    degrees[i],
                    k,
                    neighbours_i[j].0,
                    neighbours_i[j].1
                );
                nb_deleted += 1;
                if k <= nb_edges_check {
                    nb_found += 1;
                } else {
                    //            break;
                    continue;
                }
            }
        } // end loop on all potential edges
        log::debug!(
            "node {}, nb found edge deleted : {}, deleted : {}",
            i,
            nb_found,
            nb_deleted
        );
        //
        let precision: f64;
        let precision_paper: f64;
        {
            // paper block
            precision_paper = nb_found as f64 / nb_edges_check.min(degrees[i]) as f64;
            log::debug!("inserting in histogram_paper : {:.3e}", precision_paper);
            histogram_paper.write().unwrap().insert(precision_paper);
        }
        //
        {
            // our precision, we depart from paper
            // if there is no deleted edge, how can this node contribute to anything??
            if nb_deleted > 0 {
                precision = nb_found as f64 / nb_deleted.min(nb_edges_check).min(degrees[i]) as f64;
                log::debug!("inserting in histogram : {:.3e}", precision);
                histogram.write().unwrap().insert(precision);
            } else {
                precision = -1.; // here we depart from paper which returns 0.
            }
        }
        //
        (precision_paper, precision)
    };
    //
    // paper precision
    //
    let nodes_precision_paper: Vec<(usize, f64)> =
        selected_nodes.iter().map(|i| (*i, f(*i).0)).collect();
    let precision_paper: Vec<f64> = nodes_precision_paper.iter().map(|t| t.1).collect();
    let mean_precision_paper: f64 =
        precision_paper.iter().sum::<f64>() / precision_paper.len() as f64;
    let mut sigma_precision_paper = precision_paper.iter().fold(0., |acc, x| {
        acc + (x - mean_precision_paper) * (x - mean_precision_paper)
    });
    sigma_precision_paper /= precision_paper.len() as f64;
    sigma_precision_paper = (sigma_precision_paper / precision_paper.len() as f64).sqrt();
    //
    // our precision
    //
    let nodes_precision: Vec<(usize, f64)> = selected_nodes
        .iter()
        .map(|i| (*i, f(*i).1))
        .filter(|p| p.1 >= 0.)
        .collect();
    //
    let precision: Vec<f64> = nodes_precision.iter().map(|t| t.1).collect();
    let mean_precision: f64 = precision.iter().sum::<f64>() / precision.len() as f64;
    let mut sigma_precision = precision.iter().fold(0., |acc, x| {
        acc + (x - mean_precision) * (x - mean_precision)
    });
    sigma_precision /= precision.len() as f64;
    sigma_precision = (sigma_precision / precision.len() as f64).sqrt();
    //
    let count = histogram.read().unwrap().count();
    log::info!("==============\n");
    log::info!("results");
    log::info!(
        "nb nodes examined : {:?}, nb nodes with statistics : {}, nb nodes with statistics : {}",
        nb_sampled,
        histogram_paper.read().unwrap().count(),
        count
    );
    //
    if count > 0 {
        let nbslot = 40;
        println!("\n\n vcmpr quantiles");
        println!("quantile         vcmpr(paper)    vcmpr(ours)");
        for i in 0..=nbslot {
            let q = i as f64 / nbslot as f64;
            println!(
                "{:.3e}        {:.3e}          {:.3e}",
                q,
                histogram_paper
                    .read()
                    .unwrap()
                    .query(q)
                    .unwrap_or((0, 0.))
                    .1,
                histogram.read().unwrap().query(q).unwrap_or((0, 0.)).1,
            );
        }
        log::info!(
            "paper precision @{}: {:.3e}, std deviation : {:.3e}",
            nb_edges_check,
            mean_precision_paper,
            sigma_precision_paper
        );
        log::info!("\n");
        log::info!(
            "precision @{}: {:.3e}, std deviation : {:.3e}",
            nb_edges_check,
            mean_precision,
            sigma_precision
        );
    } else {
        log::error!("empty quantiles");
    }
    // get correlation between precision and degrees
    log::info!("\n");
    log::info!("correlation");
    let selected_degrees: Vec<f64> = nodes_precision.iter().map(|t| t.0 as f64).collect();
    let rho = pearson_cor(&selected_degrees, &precision);
    log::info!("precision , degree correlation : {:.3e}", rho);
    let selected_degrees_papers: Vec<f64> =
        nodes_precision_paper.iter().map(|t| t.0 as f64).collect();
    let rho = pearson_cor(&selected_degrees_papers, &precision_paper);
    log::info!("precision_paper , degree correlation : {:.3e}", rho);
    //
    log::info!(
        "\n estimate_vcmpr sys time(s) {:.2e} cpu time(s) {:.2e}",
        sys_start.elapsed().unwrap().as_secs() as f64,
        cpu_start.elapsed().as_secs()
    );
    //
    return;
} // end of estimate_vcmpr

//
//
#[cfg_attr(doc, katexit::katexit)]
///
/// This function is inspired by the paper:  
/// Link prediction using low-dimensional node embeddings:The measurement problem (2024)
/// See [vcmpr](https://www.pnas.org/doi/10.1073/pnas.2312527121)
/// It tries to remedy to certain interpretation difficulties related to the paper discussed here: [linkauc](https://github.com/jean-pierreBoth/Linkauc)
///
/// 1. Method
///
/// We compute a vertex centric measure of performance, but instead of using a precision@k (for an arbitrary k) we take benefit
/// of the sorting of all distances around a node to compute an AUC for each node examined.
/// We note :
/// - $nbnodes$ the number of nodes in the graph:
/// - $n$ the node we are studying
/// - $d$ the degree of $n$.
/// - $de$ the number deleted edge having $n$ as extremity.  
///
///  So the node has $d$ - $de$ true edges after edge deletion (they are the train edges) and $nbnodes$ - $d$ + $de$ potential edges.
///
/// - We sort in increasing order all nodes distances to $n$ in the embedding (dot product for Hope or Jaccard for sketching)
/// - We parse this array, noting $j$  the current position in the parsing loop.
///   If j corresponds to a true (train) edge we increment the counter $k$ of true edges encountered up to j
///   If j corresponds to a deleted (test) neighbour edge: the question we must answer is :  
///     what is the probability that it has a smaller distance to our reference node $n$ than a random edge?    
///   As the array is sorted, the response is just the number of indexes greater than j that do not correspond to a true edge so it is
///     $$ (nbnodes - j -(d  - de - k))/ (nbnodes -1 - d  + de) $$
///   this fraction is linearly decreasing as j increases.    
///   If $j=nbnodes$ and we have a deleted edge then this edge is the last we get $k = d - de$ and this last edge contributes 0.
///   Averging over k we get the centric auc of n and finally averaging over 2000 nodes $n$ we get an estimate of centric auc over the graph.
///
/// 2. Outputs:
///  The function outputs:
/// -  mean centric auc and standard deviation
/// -  degrees quantiles
/// -  centric auc quantiles to check for high variations dependind on points.
/// -  correlation coefficients between degrees and centric auc.
/// -  time spent in the function (as the sorting of larges arrays cost cpu time)
///
pub fn estimate_centric_auc<F, G, E>(
    csmat: &CsMatI<F, usize>,
    _nbiter: usize,
    delete_proba: f64,
    symetric: bool,
    embedder: &(dyn Fn(TriMatI<F, usize>) -> E + Sync),
) where
    F: Default + Copy + std::marker::Sync,
    G: std::fmt::Debug,
    E: EmbeddedT<G> + std::marker::Sync,
{
    log::info!("===================================================");
    log::info!("in estimate_centric_auc symetric mode: {:?}\n\n", symetric);
    log::info!("===================================================");
    //
    let cpu_start = ProcessTime::now();
    let sys_start = SystemTime::now();
    // we begin by a degree estimation
    let degrees = csmat.degrees();
    let mut mean_degree: f64 = 0.;
    let mut max_degree: usize = 0;
    for d in &degrees {
        mean_degree += (*d) as f64;
        max_degree = (*d).max(max_degree);
    }
    mean_degree /= degrees.len() as f64;
    log::info!(
        "mean degree : {:.3e}, max_degree : {}",
        mean_degree,
        max_degree
    );
    //
    let mut degree_histogram = CKMS::<u32>::new(0.001);
    for d in &degrees {
        degree_histogram.insert((*d).try_into().unwrap());
    }
    let nbslot = 40;
    let mut qs = Vec::<f64>::with_capacity(30);
    for i in 1..nbslot {
        let q = i as f64 / nbslot as f64;
        qs.push(q);
    }
    qs.push(0.99);
    qs.push(0.999);
    println!("\n quantiles degrees of graph \n");
    println!("quantiles, degrees");
    for q in qs {
        println!(" {:.3e},   {}", q, degree_histogram.query(q).unwrap().1);
    }
    println!("\n");
    //
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(456231);
    // split edges into kept and discarded
    let (trimat, deleted_edges) = filter_csmat(csmat, delete_proba, symetric, &mut rng);
    // need to store trimat index before move to embedding
    let mut trimat_set = HashSet::<(usize, usize)>::with_capacity(trimat.nnz());
    for triplet in trimat.triplet_iter() {
        trimat_set.insert((triplet.1 .0, triplet.1 .1));
    }
    log::debug!(
        "trimat_set size : {} nb deleted edges : {}",
        trimat_set.len(),
        deleted_edges.len()
    );
    //
    // embedder (passed as a closure)
    //
    let embedded = &embedder(trimat);
    if !embedded.is_symetric() {
        log::warn!("method estimate_vcmpr only possible for symetric embeddings");
    }
    let nb_nodes = csmat.shape().0;
    let nb_to_sample = 2000;
    // sample nodes according to paper algo or mine
    let uniform: bool = true;
    log::info!("sampling uniform : {:?}", uniform);
    log::info!("==================================");
    //
    let selected_nodes = if uniform {
        sample_nodes_uniform(csmat, nb_to_sample)
    } else {
        sample_nodes_by_degrees(csmat, nb_to_sample)
    };
    //
    // select nodes we will test
    let nb_sampled = selected_nodes.len();
    log::info!("estimate_vcmpr nb nodes sampled : {}", nb_sampled);
    // This function returns integral auc score on deleted edge
    let count_deleted = |i: usize| -> usize {
        let mut nb_deleted = 0;
        for edge in &deleted_edges {
            if edge.0 == i {
                nb_deleted += 1;
            }
        }
        nb_deleted
    };
    //
    let compute_e_auc = |i: usize| -> Option<f64> {
        // how good is the embedding of node i
        // sort edges by decreasing length
        let mut neighbours_i: Vec<(usize, f64)> = (0..nb_nodes)
            .into_par_iter()
            .map(|j| (j, embedded.get_noderank_distance(i, j)))
            .collect();
        // get neighbours sorted in increasing distance (small distance are most probable edges!)
        neighbours_i.sort_unstable_by(|a, b| (a.1).partial_cmp(&b.1).unwrap());
        //
        assert!(
            neighbours_i.first().as_ref().unwrap().1 <= neighbours_i.last().as_ref().unwrap().1
        );
        //
        if i <= 50 {
            for k in 0..20 {
                log::debug!(
                    "neighbours_i beginning, i : {} j : {}, d = {:.3e}",
                    i,
                    neighbours_i[k].0,
                    neighbours_i[k].1
                );
            }
        }
        // how many  deleted edges from this node
        let nb_deleted = count_deleted(i);
        if nb_deleted == 0 {
            return None; // if no edge deleted , cannot contribute to auc
        }
        let nb_potential_edges = nb_nodes - 1 - degrees[i] + nb_deleted;
        assert!(nb_potential_edges > 0);
        let mut nb_found_true = 0;
        let mut found_ranks = Vec::<usize>::with_capacity(degrees[i]);
        for j in 1..neighbours_i.len() {
            // we bypass existing edges
            if trimat_set.contains(&(i, neighbours_i[j].0)) {
                nb_found_true += 1;
                continue;
            }
            // we know we have a potential edge
            if deleted_edges.contains(&(i, neighbours_i[j].0)) {
                log::debug!(
                    " node {}, degree : {}, edge rank : {}, neighbour : {},, dist : {:.3e}",
                    i,
                    degrees[i],
                    j,
                    neighbours_i[j].0,
                    neighbours_i[j].1
                );
                assert!(nb_nodes - degrees[i] + nb_deleted >= j - nb_found_true);
                found_ranks.push(nb_nodes - j - (degrees[i] - nb_deleted - nb_found_true));
            }
        } // end loop on all potential edges
        log::debug!(
            "found edge deleted for node : {}, deleted : {}",
            i,
            found_ranks.len()
        );
        // compute auc
        let mut e_auc: f64 = 0.;
        for m in &found_ranks {
            e_auc = *m as f64 / nb_potential_edges as f64;
        }
        //
        log::debug!("node : {}, degree : {}, auc : {:.3e}", i, degrees[i], e_auc);
        //
        Some(e_auc)
    };
    //
    let nodes_e_auc: Vec<(usize, f64)> = selected_nodes
        .iter()
        .map(|i| (i, compute_e_auc(*i)))
        .filter(|node_opt| node_opt.1.is_some())
        .map(|node_opt| (*node_opt.0, node_opt.1.unwrap()))
        .collect();
    //
    log::info!(
        "nb nodes examined : {:?}, nb nodes with a deleted edge : {}",
        selected_nodes.len(),
        nodes_e_auc.len()
    );
    // get degrees and auc for correlation output
    let selected_degrees: Vec<f64> = nodes_e_auc.iter().map(|t| degrees[t.0] as f64).collect();
    let selected_auc: Vec<f64> = nodes_e_auc.iter().map(|t| t.1).collect();
    let mean_auc: f64 = selected_auc.iter().sum::<f64>() / selected_auc.len() as f64;
    let mut sigma_auc = selected_auc
        .iter()
        .fold(0., |acc, x| acc + (x - mean_auc) * (x - mean_auc));
    sigma_auc /= selected_auc.len() as f64;
    sigma_auc = (sigma_auc / selected_auc.len() as f64).sqrt();
    //
    // dump histogram
    //
    if selected_auc.len() > 0 {
        let mut histogram = CKMS::<f64>::new(0.01);
        for f in &selected_auc {
            histogram.insert(*f);
        }
        println!("\n centric auc quantiles");
        println!("quantile, centric auc");
        for i in 0..=20 {
            let q = i as f64 / 20.;
            println!("{:.3e},   {:.3e}", q, histogram.query(q).unwrap().1);
        }
        log::info!(
            "average e_auc : {:.3e}, std deviation : {:.3e}",
            mean_auc,
            sigma_auc
        );
    }
    //
    let rho = pearson_cor::<f64>(&selected_degrees, &selected_auc);
    log::info!("\n centric auc , degree correlation : {:.3e}", rho);
    log::info!(
        "\n estimate_centric_auc sys time(s) {:.2e} cpu time(s) {:.2e}",
        sys_start.elapsed().unwrap().as_secs() as f64,
        cpu_start.elapsed().as_secs()
    );
    //
    return;
} // end of estimate_centric_auc

//================================================================================================================

#[cfg(test)]
mod tests {

    //    cargo test csv  -- --nocapture
    //    cargo test validation::link::tests::test_name -- --nocapture
    //    RUST_LOG=graphite::validation::link=TRACE cargo test link* -- --nocapture

    use super::*;

    use crate::io::csv::*;

    use crate::prelude::*;

    use sprs::TriMatI;

    use crate::embed::nodesketch::nodesketchsym::*;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[allow(unused)]
    // makes a (symetric) nodesketch Embedded to be sent to precision computations
    fn nodesketch_get_embedded(trimat: TriMatI<f64, usize>) -> Embedded<usize> {
        let sketch_size = 300;
        let decay = 0.2;
        let nb_iter = 10;
        let symetric = true;
        let parallel = true;
        let params = NodeSketchParams {
            sketch_size,
            decay,
            nb_iter,
            symetric,
            parallel,
        };
        log::debug!(" embedding parameters : {:?}", params);
        // now we embed
        let mut nodesketch = NodeSketch::new(params, trimat);
        let embed_res = nodesketch.embed();
        embed_res.unwrap()
    } // end nodesketch_get_embedded

    #[allow(unused)]
    fn nodesketchasym_get_embedded(trimat: TriMatI<f64, usize>) -> EmbeddedAsym<usize> {
        let sketch_size = 900;
        let decay = 0.2;
        let nb_iter = 5;
        let symetric = false;
        let parallel = true;
        let params = NodeSketchParams {
            sketch_size,
            decay,
            nb_iter,
            symetric,
            parallel,
        };
        log::debug!(" embedding parameters : {:?}", params);
        // now we embed
        let mut nodesketch = NodeSketchAsym::new(params, trimat);
        let embed_res = nodesketch.embed();
        embed_res.unwrap()
    } // end nodesketchasym_get_embedded

    #[test]
    fn test_link_precision_nodesketch_lesmiserables() {
        //
        log_init_test();
        //
        log::debug!("in link.rs test_nodesketch_lesmiserables");
        let path = std::path::Path::new(crate::DATADIR)
            .join("moreno_lesmis")
            .join("out.moreno_lesmis_lesmis");
        log::info!(
            "\n\n test_nodesketch_lesmiserables, loading file {:?}",
            path
        );
        let res = csv_to_trimat::<f64>(&path, false, b' ');
        if res.is_err() {
            log::error!("test_nodesketch_lesmiserables failed in csv_to_trimat");
            assert_eq!(1, 0);
        } else {
            let trimat_indexed = res.unwrap();
            let csrmat: CsMatI<f64, usize> = trimat_indexed.0.to_csr();
            let symetric = true;
            let precision = estimate_precision(&csrmat, 5, 0.1, symetric, &nodesketch_get_embedded);
            log::trace!("precision : {:?}", precision.0);
            log::trace!("recall : {:?}", precision.1);
        };
    } // end of test_link_precision_nodesketch_lesmiserables

    #[test]
    fn test_link_auc_nodesketch_lesmiserables() {
        //
        log_init_test();
        //
        log::debug!("in link.rs test_nodesketch_lesmiserables");
        let path = std::path::Path::new(crate::DATADIR)
            .join("moreno_lesmis")
            .join("out.moreno_lesmis_lesmis");
        log::info!(
            "\n\n test_nodesketch_lesmiserables, loading file {:?}",
            path
        );
        let res = csv_to_trimat::<f64>(&path, false, b' ');
        if res.is_err() {
            log::error!("test_nodesketch_lesmiserables failed in csv_to_trimat");
            assert_eq!(1, 0);
        } else {
            let trimat_indexed = res.unwrap();
            let csrmat: CsMatI<f64, usize> = trimat_indexed.0.to_csr();
            let symetric = true;
            let auc = estimate_auc(&csrmat, 5, 0.1, symetric, &nodesketch_get_embedded);
            log::info!("auc : {:?}", auc);
        };
    } // end of test_link_auc_nodesketch_lesmiserables

    // We can always treat a symetric as an asymetric one. Check results
    #[test]
    fn test_link_auc_nodesketchasym_lesmiserables() {
        //
        log_init_test();
        //
        log::debug!("in link.rs test_nodesketchasym_lesmiserables");
        let path = std::path::Path::new(crate::DATADIR)
            .join("moreno_lesmis")
            .join("out.moreno_lesmis_lesmis");
        log::info!(
            "\n\n test_nodesketch_lesmiserables, loading file {:?}",
            path
        );
        // we keep directed as false to get symetrization done in csv_to_trimat!!
        let res = csv_to_trimat::<f64>(&path, false, b' ');
        if res.is_err() {
            log::error!("test_nodesketchasym_lesmiserables failed in csv_to_trimat");
            assert_eq!(1, 0);
        } else {
            let trimat_indexed = res.unwrap();
            let csrmat: CsMatI<f64, usize> = trimat_indexed.0.to_csr();
            let symetric = false;
            let auc = estimate_auc(&csrmat, 5, 0.1, symetric, &nodesketchasym_get_embedded);
            log::info!("auc : {:?}", auc);
        };
    } // end of test_link_auc_nodesketchasym_lesmiserables

    // ============================  auc testing for hope ==========================  //

    // functon to pass to auc methods
    #[allow(unused)]
    fn hope_ada_get_embedded(trimat: TriMatI<f64, usize>) -> EmbeddedAsym<f64> {
        let nb_iter = 5;
        let hope_m = HopeMode::ADA;
        let decay_f = 0.05;
        //        let range_m = RangeApproxMode::RANK(RangeRank::new(70, 2));
        let range_m = RangeApproxMode::EPSIL(RangePrecision::new(0.1, 10, 300));
        let params = HopeParams::new(hope_m, range_m, decay_f);
        // now we embed
        let mut hope = Hope::new(params, trimat);
        //
        let embed_res = hope.embed();
        embed_res.unwrap()
    } // end hope_ada_get_embedded

    #[test]
    fn test_link_auc_hope_ada_lesmiserables() {
        //
        log_init_test();
        //
        log::debug!("in link.rs test_link_auc_hope_ada_lesmiserables");
        let path = std::path::Path::new(crate::DATADIR)
            .join("moreno_lesmis")
            .join("out.moreno_lesmis_lesmis");
        log::info!(
            "\n\n test_nodesketch_lesmiserables, loading file {:?}",
            path
        );
        // we keep directed as false to get symetrization done in csv_to_trimat!!
        let res = csv_to_trimat::<f64>(&path, false, b' ');
        if res.is_err() {
            log::error!("test_nodesketchasym_lesmiserables failed in csv_to_trimat");
            assert_eq!(1, 0);
        } else {
            let trimat_indexed = res.unwrap();
            let csrmat: CsMatI<f64, usize> = trimat_indexed.0.to_csr();
            let symetric = true;
            let auc = estimate_auc(&csrmat, 5, 0.1, symetric, &hope_ada_get_embedded);
            log::info!("auc : {:?}", auc);
        }
    } // end of test_link_auc_hope_ada_lesmiserables
} // end of mod tests
