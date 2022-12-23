//! a simple link prediction to test Embeddeds



/// The scoring function for similarity is based upon the distance relative to the Embedded being tested
/// Jaccard for nodesketch and L2 for Atp
/// We first implement precision measure as described in 
///       - Link Prediction in complex Networks : A survey
///             Lü, Zhou. Physica 2011
///
///  Another reference is :
///       - Link prediction problem for social networks
///             David Liben-Nowell, J Kleinberg 2007

use log::{log_enabled};

use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;
use rand::distributions::{Uniform, Distribution};

use std::collections::{HashSet, HashMap};
use indexmap::IndexSet;

use sprs::{TriMatI, CsMatI};

use rayon::prelude::*;

use crate::embedding::{EmbeddedT};
use crate::embed::tools::{degrees::*, edge::Edge, edge::IN, edge::OUT};

pub enum ValidationMode {
    NODELABEL,
}


// ======================================================================================



// filter out edge with proba delete_proba
// The return a tuple containing the matrix of the graph with some edge deleted and a perfect hash on deleted edges
fn filter_csmat<F>(csrmat : &CsMatI<F, usize>, delete_proba : f64, symetric : bool, rng : &mut Xoshiro256PlusPlus) -> (TriMatI::<F, usize>, IndexSet<(usize,usize)>)
    where F: Default + Copy {
    //
    assert!(delete_proba < 1. && delete_proba > 0.);
    log::debug!("filter_csmat, delete proba {}, symetric : {}", delete_proba, symetric);
    // get total in + out degree
    let nb_nodes = csrmat.rows();
    let nb_edge = csrmat.nnz();
    let mut deleted_edge = IndexSet::with_capacity((nb_edge as f64 * delete_proba).round() as usize);
 
    //
    let uniform = Uniform::<f64>::new(0., 1.);
    let mut rows = Vec::<usize>::with_capacity(nb_nodes);
    let mut cols = Vec::<usize>::with_capacity(nb_nodes);
    let mut values = Vec::<F>::with_capacity(nb_nodes);
    // we need degrees as we cannot delete edges if they are the last 
    // as we cannot train anything if nodes is disconnected
    let mut degrees = get_csmat_degrees::<F>(csrmat);
    //
    log::info!("mean in degree : {:.3e}", degrees.iter().map(|degree| degree.d_in).sum::<u32>() as f64/ nb_nodes as f64);
    //
    let mut discarded : usize = 0;
    let mut nb_isolation_not_discarded : usize = 0;
    let nb_to_discard = match symetric {
        true => { 2 * (nb_edge as f64 * delete_proba) as usize},
        false => { (nb_edge as f64 * delete_proba) as usize},
    };
    // 
    log::debug!("csrmat nb edge : {}, number to discard {}", nb_edge, nb_to_discard);

    let mut csmat_iter = csrmat.iter();
    while let Some((value,(row, col))) = csmat_iter.next() {
        let xsi = uniform.sample(rng);
        let mut discard = false;
        if xsi < delete_proba {
            // We avoid making isolated node 
            if symetric {
                // we check only for row < col as we will force coherent deletion at end of scan
                if degrees[row].d_out > 1 && degrees[col].d_in > 1 && row < col {
                    discard = true;
                }
                else {
                    nb_isolation_not_discarded += 1; 
                }
            }
            else {
                if (degrees[row].degree_out() > 1 || degrees[row].degree_in() > 0)  &&
                    (degrees[col].degree_in() > 1 || degrees[col].degree_out() > 0) && row != col {
                    discard = true;
                }
                else {
                    nb_isolation_not_discarded += 1; 
                }
            }
        }
        if !discard {
            // we keep edge in triplets
            rows.push(row);
            cols.push(col);
            values.push(*value);
        }
        else {
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
    }  // end while
    //
    log::info!(" ratio discarded = {:.3e}", discarded as f64/ (nb_edge as f64));
    log::info!(" ratio not discarded to avoid disconnected node = {:.3e}", nb_isolation_not_discarded as f64/ (nb_edge as f64));
    //
    let trimat = TriMatI::<F, usize>::from_triplets((nb_nodes,nb_nodes), rows, cols, values);
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
fn one_precision_iteration<F, G, E>(csmat : &CsMatI<F, usize>, delete_proba : f64, symetric : bool, 
            embedder : &dyn Fn(TriMatI<F, usize>) -> E, rng : &mut Xoshiro256PlusPlus) -> (f64 , f64)
            where   F: Default + Copy + std::marker::Sync ,
                    E : EmbeddedT<G> + std::marker::Sync {
    //
    // filter
    let (trimat, deleted_edges) = filter_csmat(csmat, delete_proba, symetric, rng);
    log::debug!("\n\n one_precision_iteration : nb_deleted edges : {}", deleted_edges.len());
    // We construct the list of edges not in reduced graph
    let nb_nodes = csmat.shape().0;
    let max_degree = csmat.degrees().into_iter().max().unwrap();
    let mut embedded_edges = Vec::<Edge>::with_capacity(nb_nodes*nb_nodes);
    // filter out self loops and edges in trimat so we keep all edges except the train set
    let f_i = |i : usize| -> Vec<Edge> {
        let mut edges_i = Vec::<Edge>::with_capacity(max_degree);
        for j in 0..nb_nodes {
            if j!= i && (&trimat).find_locations(i,j).len() == 0usize {
                edges_i.push(Edge{0:i, 1:j, 2: 0f64});
            }
        }
        edges_i                       
    };
    let mut row_embedded : Vec<Vec<Edge>> = (0..nb_nodes).into_par_iter().map(|i| f_i(i)).collect();
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
    log::debug!("one_precision_iteration: complete graph minus deleted edges in  : {}", embedded_edges.len());
    log::debug!("nb embedded edges : {}, smallest : {:?}, largest : {:?}", embedded_edges.len(),embedded_edges[0], embedded_edges[embedded_edges.len()-1]);
    //
    //  TODO parametrize? we give precision at 95% of deleted
    let retrieved: usize = (0.95 * deleted_edges.len() as f64) as usize;
    embedded_edges.truncate(retrieved);
    log::debug!("\n keeping {}, smallest non diagonal edges remaining {:?} ... {:?}", retrieved, embedded_edges[0], embedded_edges[embedded_edges.len()-1]);
    // how many deleted edges (and so not learned in the embedding) are in the edges list with smallest distance
    let mut h_edges = HashMap::<(usize,usize), f64>::with_capacity(embedded_edges.len());
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
    let recall = nb_in as f64/ deleted_edges.len() as f64;
    let precision = nb_in as f64/ retrieved as f64;
    assert!(precision <= 1.);
    let f1 : f64 = 2. * recall * precision / (precision+recall);
    log::debug!("one_precision_iteration : precison {:.3e}, recall : {:.3e} nb_deleted : {}, nb_retrieved : {}", precision, recall, deleted_edges.len(), retrieved);
    log::info!("f1 = {:.3e}", f1);    
    //
    (precision,recall)
} // end of one_precision_iteration



/// estimate precision on nbiter iterations and edge deletion probability.
/// 
/// Precision estimation is a costly operation as it requires computing the whole (nbnode, nbnode) matrix similarity
/// between embedded nodes and sorting them to compare with the deleted edges. So precision uses a comparison of each deleted edge
/// with most probables inexistant edges.  
/// 
/// AUC instead samples, for each deleted edge, a random inexistant edge from the original graph and compute its probability in the embedded graph
/// count number of times the deleted edge is more probable than the deleted.
/// 
pub fn estimate_precision<F, G, E>(csmat : &CsMatI<F, usize>, nbiter : usize, delete_proba : f64, symetric : bool, 
            embedder : & dyn Fn(TriMatI<F, usize>) -> E) -> (Vec<f64>, Vec<f64>) 
            where  F : Default + Copy + Sync,  E : EmbeddedT<G> + std::marker::Sync {
    //
    log::debug!("in estimate_precision delete_proba : {:.3e},  symetric : {}", delete_proba, symetric);
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
        let (iter_prec, iter_recall) = one_precision_iteration(csmat, delete_proba,  symetric, embedder, &mut rngs[i]);
        precision.push(iter_prec);
        recall.push(iter_recall);
    }
    //
    let mean_recall : f64 = recall.iter().sum::<f64>() / recall.len() as f64;
    let mean_precision : f64 = precision.iter().sum::<f64>() / recall.len() as f64;
    log::info!("\n estimate_precision   mean recall : {:.3e}, mean_precision : {:.3e}", mean_recall, mean_precision);
    //
    (precision,recall)
} // end of estimate_precision






/// type G is necessary beccause we embed in possibly different type than F. (for example in Array<usize> with nodesketch)
fn one_auc_iteration<F, G, E>(csmat : &CsMatI<F, usize>, delete_proba : f64, symetric : bool, 
            embedder : &dyn Fn(TriMatI<F, usize>) -> E, mut rng: Xoshiro256PlusPlus) -> f64
            where   F : Default + Copy + std::marker::Sync,
                    G : std::fmt::Debug,
                    E : EmbeddedT<G> + std::marker::Sync {
        //
    let nb_sample = 10000;
    let mut nb_dist_equality: usize = 0;
    log::debug!("\n\n in one_auc_iteration nb_sample : {:?}, delete_proba : {:.3e}, symetric {:?}", nb_sample, delete_proba, symetric);
    //
    // sample nb_sample 2-uples of edges one from the deleted edge and one inexistent (not in csmat)
    // count the number of times the distance of the first is less than the second.
    // filter
    let (trimat, deleted_edges) = filter_csmat(csmat, delete_proba, symetric, &mut rng);
    // need to store trimat index before move to embedding
    let mut trimat_set = HashSet::<(usize,usize)>::with_capacity(trimat.nnz());
    for triplet in trimat.triplet_iter() {
        trimat_set.insert((triplet.1.0, triplet.1.1));
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
        let del_edge = deleted_edges.get_index(del_uniform.sample(&mut rng)).unwrap();
        let no_edge = loop {
            let i = node_random.sample(&mut rng);
            let j = node_random.sample(&mut rng);
            if i != j && !trimat_set.contains(&(i,j)) && deleted_edges.get_index_of(&(i,j)).is_none() {
                // edge (i,j) not on diagonal and neither in trimat set neither in deleted_edges, so inexistent edge
                break (i,j);
            }
            if !symetric && i != j && !trimat_set.contains(&(j,i)) && deleted_edges.get_index_of(&(j,i)).is_none() {
                // edge (i,j) not on diagonal and neither in trimat set neither in deleted_edges, so inexistent edge
                break (j,i);
            }
        };
        let dist_del_edge = embedded.get_noderank_distance(del_edge.0,del_edge.1);
        let dist_no_edge = embedded.get_noderank_distance(no_edge.0,no_edge.1);
        // debug stuff
        if log_enabled!(log::Level::Trace) {
            log::debug!("distance between deleted edge nodes {} and {} : {:.3e}", del_edge.0, del_edge.1, dist_del_edge);
            log::debug!("distance between no edge nodes {} and {} : {:.3e}",no_edge.0, no_edge.1, dist_no_edge);
//            log::trace!(" dump node del_edge.0, {:?} : {:?}", no_edge.0, embedded.get_embedded_node(del_edge.0, 0));
//            log::trace!(" dump node del_edge.1, {:?} : {:?}", no_edge.1, embedded.get_embedded_node(del_edge.1, 1));        
            if dist_no_edge <  dist_del_edge {
                log::debug!(" node rank out del_edge.0, {:?} : {:?}", del_edge.0, embedded.get_embedded_node(del_edge.0, OUT));
                log::debug!(" node rank out del_edge.1, {:?} : {:?}", del_edge.1, embedded.get_embedded_node(del_edge.1, OUT));
                log::debug!(" node rank out no_edge.0, {:?} : {:?}", no_edge.0, embedded.get_embedded_node(no_edge.0, OUT));
                log::debug!(" node rank out no_edge.1, {:?} : {:?}", no_edge.1, embedded.get_embedded_node(no_edge.1, OUT));                
                if !symetric {
                    log::debug!(" node rank in del_edge.0, {:?} : {:?}", del_edge.0, embedded.get_embedded_node(del_edge.0, IN));
                    log::debug!(" node rank in del_edge.1, {:?} : {:?}", del_edge.1, embedded.get_embedded_node(del_edge.1, IN));
                    log::debug!(" node rank in no_edge.0, {:?} : {:?}", no_edge.0, embedded.get_embedded_node(no_edge.0, IN));
                    log::debug!(" node rank in no_edge.1, {:?} : {:?}", no_edge.1, embedded.get_embedded_node(no_edge.1, IN)); 
                }
            }
        }
        // end debug stuff
        if dist_del_edge < dist_no_edge {
            good += 1.;
        }
        else if dist_del_edge <= dist_no_edge {
            good += 0.5;
            nb_dist_equality += 1;
        }
    }
    let auc = good / nb_sample as f64;
    log::info!(" auc = {:3.e} nb dist equality : {}", auc, nb_dist_equality);
    //
    return auc;
} // end of one_auc_iteration


/// estimate AUC as described in Link Prediction in complex Networks : A survey
///             Lü, Zhou. Physica 2011
/// 
/// type G is necessary beccause we embed in a possibly different type than F. (for example in Array\<usize\> with nodesketch)

pub fn estimate_auc<F, G, E>(csmat : &CsMatI<F, usize>, nbiter : usize, delete_proba : f64, symetric : bool, 
            embedder : &(dyn Fn(TriMatI<F, usize>) -> E + Sync)) -> Vec<f64>
    where   F : Default + Copy + std::marker::Sync,
            G : std::fmt::Debug,
            E : EmbeddedT<G> + std::marker::Sync {
        //
        log::debug!("in estimate_auc");
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
            auc = (0..nbiter).into_par_iter().map(|i| one_auc_iteration(csmat, delete_proba,  symetric, embedder, rngs[i].clone())).collect();
        }
        else {
            for i in 0..nbiter {
                let iter_auc = one_auc_iteration(csmat, delete_proba,  symetric, embedder, rngs[i].clone());
                auc.push(iter_auc);
            }
        }
        let mean_auc : f64 = auc.iter().sum::<f64>() / (auc.len() as f64);
        let sigma2 =  auc.iter().fold(0.0f64, | var : f64, x |  var + (*x - mean_auc) * (*x - mean_auc))/ (auc.len() as f64);
        log::info!("estimate_auc : mean auc : {:.3e}, std dev : {:.3e}", mean_auc, (sigma2/ (auc.len() as f64)).sqrt());
        log::debug!("exiting estimate_auc");
        //
        auc
} // end of estimate_auc



//================================================================================================================

#[cfg(test)]
mod tests {



//    cargo test csv  -- --nocapture
//    cargo test validation::link::tests::test_name -- --nocapture
//    RUST_LOG=graphite::validation::link=TRACE cargo test link* -- --nocapture

    use super::*;

    use crate::io::csv::*;

    use crate::prelude::*;

    use sprs::{TriMatI};

    use crate::embed::nodesketch::{nodesketchsym::*};

    
    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }  

    #[allow(unused)]
    // makes a (symetric) nodesketch Embedded to be sent to precision computations
    fn nodesketch_get_embedded(trimat : TriMatI<f64, usize>) -> Embedded<usize> {
        let sketch_size = 300;
        let decay = 0.2;
        let nb_iter = 10;
        let symetric = true;
        let parallel = true;
        let params = NodeSketchParams{sketch_size, decay, nb_iter, symetric, parallel};
        log::debug!(" embedding parameters : {:?}", params);
        // now we embed
        let mut nodesketch = NodeSketch::new(params, trimat);
        let embed_res = nodesketch.embed();
        embed_res.unwrap()
    } // end nodesketch_get_embedded


    #[allow(unused)]
    fn nodesketchasym_get_embedded(trimat : TriMatI<f64, usize>) -> EmbeddedAsym<usize> {
        let sketch_size = 900;
        let decay = 0.2;
        let nb_iter = 5;
        let symetric = false;
        let parallel = true;
        let params = NodeSketchParams{sketch_size, decay, nb_iter, symetric, parallel};
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
        let path = std::path::Path::new(crate::DATADIR).join("moreno_lesmis").join("out.moreno_lesmis_lesmis");
        log::info!("\n\n test_nodesketch_lesmiserables, loading file {:?}", path);
        let res = csv_to_trimat::<f64>(&path, false, b' ');
        if res.is_err() {
            log::error!("test_nodesketch_lesmiserables failed in csv_to_trimat");
            assert_eq!(1, 0);
        }
        else {
            let trimat_indexed = res.unwrap();
            let csrmat  : CsMatI<f64, usize> = trimat_indexed.0.to_csr();
            let symetric = true;
            let precision = estimate_precision(&csrmat, 5, 0.1, symetric, &nodesketch_get_embedded);
            log::trace!("precision : {:?}", precision.0);
            log::trace!("recall : {:?}", precision.1);
        };
    }  // end of test_link_precision_nodesketch_lesmiserables



    #[test]
    fn test_link_auc_nodesketch_lesmiserables() {
        //
        log_init_test();
        //
        log::debug!("in link.rs test_nodesketch_lesmiserables");
        let path = std::path::Path::new(crate::DATADIR).join("moreno_lesmis").join("out.moreno_lesmis_lesmis");
        log::info!("\n\n test_nodesketch_lesmiserables, loading file {:?}", path);
        let res = csv_to_trimat::<f64>(&path, false, b' ');
        if res.is_err() {
            log::error!("test_nodesketch_lesmiserables failed in csv_to_trimat");
            assert_eq!(1, 0);
        }
        else {
            let trimat_indexed = res.unwrap();
            let csrmat  : CsMatI<f64, usize> = trimat_indexed.0.to_csr();
            let symetric = true;
            let auc = estimate_auc(&csrmat, 5, 0.1, symetric, &nodesketch_get_embedded);
            log::info!("auc : {:?}", auc);
        };
    }  // end of test_link_auc_nodesketch_lesmiserables



    // We can always treat a symetric as an asymetric one. Check results
    #[test]
    fn test_link_auc_nodesketchasym_lesmiserables() {
        //
        log_init_test();
        //
        log::debug!("in link.rs test_nodesketchasym_lesmiserables");
        let path = std::path::Path::new(crate::DATADIR).join("moreno_lesmis").join("out.moreno_lesmis_lesmis");
        log::info!("\n\n test_nodesketch_lesmiserables, loading file {:?}", path);
        // we keep directed as false to get symetrization done in csv_to_trimat!!
        let res = csv_to_trimat::<f64>(&path, false, b' ');
        if res.is_err() {
            log::error!("test_nodesketchasym_lesmiserables failed in csv_to_trimat");
            assert_eq!(1, 0);
        }
        else {
            let trimat_indexed = res.unwrap();
            let csrmat  : CsMatI<f64, usize> = trimat_indexed.0.to_csr();
            let symetric = false;
            let auc = estimate_auc(&csrmat, 5, 0.1, symetric, &nodesketchasym_get_embedded);
            log::info!("auc : {:?}", auc);
        };
    }  // end of test_link_auc_nodesketchasym_lesmiserables



// ============================  auc testing for hope ==========================  //




    // functon to pass to auc methods
    #[allow(unused)]
    fn hope_ada_get_embedded(trimat : TriMatI<f64, usize>) -> EmbeddedAsym<f64> {
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
        let path = std::path::Path::new(crate::DATADIR).join("moreno_lesmis").join("out.moreno_lesmis_lesmis");
        log::info!("\n\n test_nodesketch_lesmiserables, loading file {:?}", path);
        // we keep directed as false to get symetrization done in csv_to_trimat!!
        let res = csv_to_trimat::<f64>(&path, false, b' ');
        if res.is_err() {
            log::error!("test_nodesketchasym_lesmiserables failed in csv_to_trimat");
            assert_eq!(1, 0);
        }
        else {
            let trimat_indexed = res.unwrap();
            let csrmat  : CsMatI<f64, usize> = trimat_indexed.0.to_csr();
            let symetric = true;
            let auc = estimate_auc(&csrmat, 5, 0.1, symetric, &hope_ada_get_embedded);
            log::info!("auc : {:?}", auc);
        }
    }  // end of test_link_auc_hope_ada_lesmiserables



    

}  // end of mod tests