//! Construct or dump a (small) graph or its matricial representation FULL or CSR from data from a csv file


//use std::fmt::{Debug};

use log::*;
use anyhow::{anyhow};

use std::collections::HashSet;

use std::fs::{OpenOptions};
use std::path::{Path};
use std::str::FromStr;

use std::io::{Read, BufReader, BufRead};

use csv::ReaderBuilder;
use num_traits::{float::*};

// recall this gives also use ndarray_linalg::{Scalar, Lapack}
use annembed::tools::svdapprox::*;

use sprs::{TriMatI, CsMat};
use indexmap::IndexSet;
//use petgraph::graph::{Graph, NodeIndex, IndexType};
use petgraph::graphmap::{GraphMap, NodeTrait};
#[allow(unused)]
use petgraph::{Directed,EdgeType};

// TODO propagate genericity on N everywhehre ?
/// maps the type N id of a node to a rank in a matrix
pub type NodeIndexation<N> = IndexSet<N>;


// count number of first lines beginning with '#' or '%'
fn get_header_size(filepath : &Path) -> anyhow::Result<usize> {
    //
    log::debug!("get_header_size");
    //
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("fn get_header_size : could not open file {:?}", filepath.as_os_str());
        println!("directed_from_csv could not open file {:?}", filepath.as_os_str());
        return Err(anyhow!("fn get_header_size : could not open file {}", filepath.display()));            
    }
    let mut file = fileres?;
    let mut nb_header_lines = 0;
    let mut c = [0];
    let mut more = true;
    while more {
        file.read_exact(&mut c)?;
        if ['#', '%'].contains(&(c[0] as char)) {
            nb_header_lines += 1;
            loop {
                file.read_exact(&mut c)?;
                if c[0] == '\n' as u8 {
                    break;
                }
            }
        }
        else {
            more = false;
            log::debug!("file has {} nb headers lines", nb_header_lines);
        }
    }
    //
    Ok(nb_header_lines)
}

// From csv file we have nodes indexed by u32, edge multiplicity as f32,  edge can be negative 
// (see bitcoin files  https://snap.stanford.edu/data/index.html)
//
// p2p-Gnutella09.txt  peer to peer directed unweighted
// ca-GrQc.txt  collaboration network undirected
//
// We must maintain an indexmap of the nodes during edge insertion

/// Ty is Directed (default) or UnDirected
/// Ix is the node and edge index type , default u32. 
/// N and E in Grap<N,E,Ty,Ix are data (weights) associated to node and edge respectively>
/// instantiate with UnDirected for undirected graph
/// 
pub fn directed_unweighted_csv_to_graph<N, Ty>(filepath : &Path, delim : u8) -> anyhow::Result<GraphMap<N, (), Ty>> 
    where   N : NodeTrait + std::hash::Hash + std::cmp::Eq + FromStr + std::fmt::Display ,
            Ty : EdgeType {
    //
    // first get number of header lines
    let nb_headers_line = get_header_size(&filepath)?;
    log::info!("directed_from_csv , got header nb lines {}", nb_headers_line);
    //
    // get rid of potential lines beginning with # or %
    // initialize a reader from filename, skip lines beginning with # or %
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("ProcessingState reload_json : reload could not open file {:?}", filepath.as_os_str());
        println!("directed_from_csv could not open file {:?}", filepath.as_os_str());
        return Err(anyhow!("directed_from_csv could not open file {}", filepath.display()));            
    }
    let mut file = fileres?;
    // skip header lines
    let mut nb_skipped = 0;
    let mut c = [0];
    loop {
        file.read_exact(&mut c)?;
        if c[0] == '\n' as u8 {
            nb_skipped += 1;
        }
        if nb_skipped  == nb_headers_line {
            break;
        }
    }
    // now we can parse records and construct a parser from current position of file
    // we already skipped headers
    let mut rdr = ReaderBuilder::new().delimiter(delim).flexible(false).has_headers(false).from_reader(file);
    //
    let nb_nodes_guess = 50_000;   // to pass as function argument
    let mut graph = GraphMap::<N, (), Ty>::with_capacity(nb_nodes_guess, 500_000);
    //
    let mut nb_record = 0;
    let mut nb_fields = 0;
    let mut node1 : N;
    let mut node2 : N;
    for result in rdr.records() {
        let record = result?;
        if log::log_enabled!(Level::Info) && nb_record <= 5 {
            log::info!("{:?}", record);
        }
        //
        if nb_record == 0 {
            nb_fields = record.len();
        }
        else {
            if record.len() != nb_fields {
                println!("non constant number of fields at record {} first record has {}",nb_record+1,  nb_fields);
                return Err(anyhow!("non constant number of fields at record {} first record has {}",nb_record+1,  nb_fields));   
            }
        }
        // we have 2 fields
        let field = record.get(0).unwrap();
        // decode into Ix type
        if let Ok(idx) = field.parse::<N>() {
            node1 = idx;
            graph.add_node(idx);
        }
        else {
            return Err(anyhow!("error decoding field 1 of record  {}",nb_record+1)); 
        }
        let field = record.get(1).unwrap();
        if let Ok(idx) = field.parse::<N>() {
            node2 = idx;
            graph.add_node(idx);
        }
        else {
            return Err(anyhow!("error decoding field 2 of record  {}",nb_record+1)); 
        }
        //
        graph.add_edge(node1, node2, ());
        //
        nb_record += 1;
        if log::log_enabled!(Level::Info) && nb_record <= 5 {
            log::info!("{:?}", record);
            log::info!(" node1 {}, node2 {}", node1, node2);
        }
        // now fill graph
    } // end of for
    log::info!("directed_unweighted_csv_to_graph read nb record : {}", nb_record);
    //
    Ok(graph)
} // end of directed_unweighted_csv_to_graph




/// load a directed/undirected  weighted/unweighted graph in csv format into a MatRepr representation.  
///  - directed must be set to true if graph is directed or if all edges are in csv file even if graph is symetric!
///  - delim is the delimiter used in the csv file necessary for csv::ReaderBuilder.
/// 
/// Returns the MatRepr field and a mapping from NodeId to a rank in matrix.
pub fn csv_to_csrmat<F:Float+FromStr>(filepath : &Path, directed : bool, delim : u8) -> anyhow::Result<(MatRepr<F>, NodeIndexation<usize>)> 
    where F: FromStr + Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + Default + Sync {
    //
    let res_csv = csv_to_trimat(filepath, directed, delim);
    if res_csv.is_ok() {
        let csrmat : CsMat<F> = res_csv.as_ref().unwrap().0.to_csr();
        log::debug!(" csrmat dims nb_rows {}, nb_cols {} ", csrmat.rows(), csrmat.cols());
        return Ok((MatRepr::from_csrmat(csrmat), res_csv.unwrap().1));
    }
    else {
        return Err(res_csv.unwrap_err());
    }
}  // end of csv_to_csrmat


/// Loads a csv file and returning a MatRepr and a reindexation of nodes to ensure that internally nodes are identified by 
/// a rank in 0..nb_nodes
///  
/// This function tests for the following delimiters [b'\t', b',', b' '] in the csv file.
/// For a symetric graph the routine expects only half of the edges are in the csv file and symterize the matrix.  
/// For an asymetric graph directed must be set to true.
pub fn csv_to_csrmat_delimiters<F:Float+FromStr>(filepath : &Path, directed : bool) -> anyhow::Result<(MatRepr<F>, NodeIndexation<usize>)> 
    where F: FromStr + Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + Sync + Default {
    //
    log::info!("\n\n csv_to_csrmat_delimiters, loading file {:?}", filepath);
    //
    let delimiters = ['\t', ',', ' ', ';'];
    //
    let mut res:  anyhow::Result<(MatRepr<F>, NodeIndexation<usize>)> = Err(anyhow!("not initializd"));
    for delim in delimiters {
        log::debug!("embedder trying reading {:?} with  delimiter {:?}", &filepath, delim);
        res = csv_to_csrmat::<F>(&filepath, directed, delim as u8);
        if res.is_err() {
            log::error!("embedder failed in csv_to_csrmat_delimiters, reading {:?}, with delimiter {:?} ", &filepath, delim);
        }
        else { return res;}
    }
    if res.is_err() {
        log::error!("error : {:?}", res.as_ref().err());
        log::error!("embedder failed in csv_to_csrmat_delimiters, reading {:?}, tested delimiers {:?}", &filepath, delimiters);
        std::process::exit(1);
    };
    //
    return res;
}  // end of csv_to_csrmat_delimiters





/// Loads a directed/undirected  weighted/unweighted graph in csv format into a TriMatI representation.  
///   
/// - directed must be set to true if graph is directed.
/// - delim is the delimiter used in the csv file necessary for csv::ReaderBuilder.
/// 
/// If there are 3 fields by record, the third is assumed to be a weight convertible type F (F morally is usize, f32 or f64)
/// Returns a 2-uple containing first the TriMatI and then the NodeIndexation remapping nodes id as given in the Csv file into (0..nb_nodes) 
/// 
pub fn csv_to_trimat<F:Float+FromStr>(filepath : &Path, directed : bool, delim : u8) -> anyhow::Result<(TriMatI<F, usize>, NodeIndexation<usize>)> 
    where F: FromStr + Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + Default {
    //
    // first get number of header lines
    let nb_headers_line = get_header_size(&filepath)?;
    log::info!("directed_from_csv , got header nb lines {}", nb_headers_line);
    //
    // as nodes num in csv files are not guaranteed to be numbered contiguously in 0..nb_nodes
    // we maintain a nodeindex. Key is nodenum as in csv file, value is node's rank in appearance order.
    // The methods get gives a rank given a num and the method get gives a num given a rank! 
    let mut nodeindex = NodeIndexation::<usize>::with_capacity(500000);
    // hset is just to detect possible edge duplicata in csv file. hashmap contains nodes id by ranks!
    let mut hset = HashSet::<(usize,usize)>::new();
    //
    // get rid of potential lines beginning with # or %
    // initialize a reader from filename, skip lines beginning with # or %
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("ProcessingState reload_json : reload could not open file {:?}", filepath.as_os_str());
        println!("directed_from_csv could not open file {:?}", filepath.as_os_str());
        return Err(anyhow!("directed_from_csv could not open file {}", filepath.display()));            
    }
    let file = fileres?;
    let mut bufreader = BufReader::new(file);
    // skip header lines
    let mut headerline = String::new();
    for _ in 0..nb_headers_line {
        bufreader.read_line(&mut headerline)?;
    }
    //
    let nb_edges_guess = 500_000;   // to pass as function argument
    let mut rows = Vec::<usize>::with_capacity(nb_edges_guess);
    let mut cols = Vec::<usize>::with_capacity(nb_edges_guess);
    let mut values = Vec::<F>::with_capacity(nb_edges_guess);
    let mut node1 : usize;   // rank id
    let mut node2 : usize;
    let mut node_id1 : usize;  // node_id as in file
    let mut node_id2 : usize;
    let mut weight : F;
    let mut rowmax : usize = 0;
    let mut colmax : usize = 0;
    let mut nb_record = 0;      // number of record loaded
    let mut num_record : usize = 0;
    let mut nb_fields = 0;
    let mut nb_self_loop = 0;
    let mut nb_nodes : usize = 0;
    let nb_warnings = 10;
    // to detect potential asymetry 
    let mut nb_potential_asymetry : usize = 0;
    let mut last_edge_inserted = (0usize,0usize);
    //
    // nodes must be numbered contiguously from 0 to nb_nodes-1 to be stored in a matrix.
    let mut rdr = ReaderBuilder::new().delimiter(delim).flexible(false).has_headers(false).from_reader(bufreader);
    for result in rdr.records() {
        num_record += 1;
        let record = result?;
        if log::log_enabled!(Level::Info) && nb_record <= 2 {
            log::info!(" record num {:?}, {:?}", nb_record, record);
        }
        //
        if nb_record == 0 {
            nb_fields = record.len();
            log::info!("nb fields = {}", nb_fields);
            if nb_fields < 2 {
                log::error!("found only one field in record, check the delimitor , got {:?} as delimitor ", delim as char);
                return Err(anyhow!("found only one field in record, check the delimitor , got {:?} as delimitor ", delim as char));
            }
        }
        else {
            if record.len() != nb_fields {
                println!("non constant number of fields at record {} first record has {}",num_record,  nb_fields);
                return Err(anyhow!("non constant number of fields at record {} first record has {}",num_record,  nb_fields));   
            }
        }
        // we have 2 or 3 fields
        let field = record.get(0).unwrap();
        // decode into Ix type
        if let Ok(node) = field.parse::<usize>() {
            node_id1 = node;
            let already = nodeindex.get_index_of(&node_id1);
            match already {
                Some(idx) => { node1 = idx},
                None             => {
                        node1 = nodeindex.insert_full(node_id1).0;
                        log::trace!("inserting node num : {}, rank : {}", node_id1, node1);
                        nb_nodes += 1;
                }
            }
            rowmax = rowmax.max(node1);
        }
        else {
            log::debug!("error decoding field 1 of record {}", num_record);
            return Err(anyhow!("error decoding field 1 of record  {}",num_record)); 
        }
        let field = record.get(1).unwrap();
        if let Ok(node) = field.parse::<usize>() {
            node_id2 = node;
            let already = nodeindex.get_index_of(&node_id2);
            match already {
                Some(idx) => { node2 = idx},
                None             => {   node2 = nodeindex.insert_full(node_id2).0;
                                        log::trace!("inserting node num : {}, rank : {}", node_id2, node2);
                                        nb_nodes += 1;
                                    }
            }            
            colmax = colmax.max(node2);
        }
        else {
            log::debug!("error decoding field 2 of record {}", num_record);
            return Err(anyhow!("error decoding field 2 of record  {}",num_record)); 
        }
        // check for self loop
        if node_id1 == node_id2 {
            nb_self_loop += 1;
            log::error!("csv_to_trimat  got diagonal term ({},{}) at record {:?} record num {}", node_id1, node_id2, record, num_record);
            continue;
        }
        //
        if !directed && !hset.insert((node1,node2)) {
            if nb_potential_asymetry <= nb_warnings {
                println!("2-uple ({:?}, {:?}) already present, record {}", node_id1, node_id2, num_record);
                log::error!("2-uple ({:?}, {:?}) already present, record {}", node_id1, node_id2, num_record);
                log::error!("last edge inserted : {:?}", last_edge_inserted);
                log::error!("record read : {:?}", record);
            }
        }
        if !directed {
            rowmax = rowmax.max(node2);
            colmax = colmax.max(node1);
        }
        if nb_fields == 3 {
            // then we read a weight
            let field = record.get(2).unwrap();
            if let Ok(w) = field.parse::<F>() {
                weight = w;
            }
            else {
                log::debug!("error decoding field 3 of record {}", nb_record+1);
                return Err(anyhow!("error decoding field 3 of record  {}",nb_record+1)); 
            }
        }
        else {
            weight = F::one();
        }
        // we store data ...
        rows.push(node1);
        cols.push(node2);
        values.push(weight);
        log::trace!("to insert : (node1, node2) : ({}, {})", node1, node2);
        nb_record += 1;
        if !directed {
            // store symetric point and warn if it was not already present
            if !hset.insert((node2,node1)) {
                nb_potential_asymetry += 1;
                if nb_potential_asymetry <= nb_warnings {
                    log::error!("undirected mode 2-uple ({:?}, {:?}) symetric edge already present, record {}", node_id2, node_id1, nb_record);
                    log::error!("last edge inserted : {:?}", last_edge_inserted);
                    log::error!("record read : {:?}", record);
                }
            }
            rows.push(node2);
            cols.push(node1);
            values.push(weight); 
        }
        last_edge_inserted.0 = node_id1;
        last_edge_inserted.1 = node_id2;
        if log::log_enabled!(Level::Info) && nb_record <= 5 {
            log::info!("{:?}", record);
            log::info!(" node1 {:?}, node2 {:?}", node1, node2);
        }
    }  // end of for result
    //
    assert_eq!(rows.len(), cols.len());
    let trimat = TriMatI::<F, usize>::from_triplets((nodeindex.len(), nodeindex.len()), rows, cols, values);
    log::debug!("trimat shape {:?}",  trimat.shape());
    assert_eq!(trimat.shape().0,nodeindex.len());
    //
    assert_eq!(nb_nodes, nodeindex.len());
    log::info!("\n\n csv file read!, nb nodes {}, nb edges loaded {}, nb_record read {}", nodeindex.len(), nb_record, num_record);
    log::info!("rowmax : {}, colmax : {}, nb_edges : {}", rowmax, colmax, trimat.nnz());
    log::info!(" nb diagonal terms filtered : {}", nb_self_loop);
    // 
    if nb_potential_asymetry > 0 {
        log::error!("\n\n csv_to_trimat : CHECK SYMETRY number of couples definded more than one time : {}", nb_potential_asymetry);
        println!("\n\n csv_to_trimat : CHECK SYMETRY number of couples definded more than one time : {}", nb_potential_asymetry);
        println!("csv_to_trimat took the first couple ")
    }
    if nb_self_loop > 0 {
        println!("nb diagonal terms filtered : {}", nb_self_loop);
    }
    // dump indexation 
    if log::log_enabled!(log::Level::Trace) {
        log::trace!("dump of indexation set");
        let mut iter = nodeindex.iter();
        let mut rank = 0;
        while let Some(id) = iter.next() {
            println!(" rank : {}, node id : {} ", rank, id);
            rank += 1;
        }
    }
    //
    Ok((trimat, nodeindex))
} // end of csv_to_trimat



/// Loads a csv file and returning a matrix representation in triplets form and a reindexation of nodes to ensure that internally nodes are identified by 
/// a rank in 0..nb_nodes
///  
/// This function tests for the following delimiters [b'\t', b',', b' '] in the csv file.
/// For a symetric graph the routine expects only half of the edges are in the csv file and symterize the matrix.  
/// For an asymetric graph directed must be set to true.
pub fn csv_to_trimat_delimiters<F:Float+FromStr>(filepath : &Path, directed : bool) -> anyhow::Result<(TriMatI<F, usize>, NodeIndexation<usize>)> 
            where F: FromStr + Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + 
                for<'r> std::ops::MulAssign<&'r F> + Default {
    //
    log::debug!("in csv_to_trimat_delimiters");
    //
    let delimiters = ['\t', ',', ' ', ';'];
    //
    let mut res :  anyhow::Result<(TriMatI<F, usize>, NodeIndexation<usize>)>  = Err(anyhow!("res not initialized"));
    for delim in delimiters {
        log::debug!("embedder trying reading {:?} with  delimiter {:?}", &filepath, delim as char);
        res = csv_to_trimat::<F>(&filepath, directed, delim as u8);
        if res.is_err() {
            log::error!("embedder failed in csv_to_trimat_delimiters, reading {:?}, trying delimiter {:?} ", &filepath, delim as char);
        }
        else { return res;}
    }
    if res.is_err() {
        log::error!("error : {:?}", res.as_ref().err());
        log::error!("embedder failed in csv_to_csrmat_delimiters, reading {:?}, tested delimiers {:?}", &filepath, delimiters);
        std::process::exit(1);
    };
    //
    return res;
}  // end of csv_to_trimat_delimiters

//========================================================================================

#[cfg(test)]

mod tests {

//    cargo test csv  -- --nocapture
//    cargo test csv::tests::test_name -- --nocapture
//    RUST_LOG=graphembed::io::csv=TRACE cargo test csv -- --nocapture



use super::*;

fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn load_undirected() {
    // 
    println!("\n\n load_undirected");
    log_init_test();

} // end test load_undirected

#[test]
fn test_directed_unweighted_csv_to_graph() {
    // We load  wiki-Vote.txt taken from Snap data directory. It is in Data directory of the crate.
    log_init_test();
    // path from where we launch cargo test
    let path = Path::new(crate::DATADIR).join("wiki-Vote.txt");
    //  # Nodes: 7115 Edges: 103689
    let header_size = get_header_size(&path);
    assert_eq!(header_size.unwrap(),4);
    println!("\n\n test_directed_unweighted_csv_to_graph data : {:?}", path);
    let graph = directed_unweighted_csv_to_graph::<u32, Directed>(&path, b'\t');
    if let Err(err) = &graph {
        eprintln!("ERROR: {}", err);
    }
    assert!(graph.is_ok());

} // end test test_directed_unweightedfrom_csv




#[test]
fn test_weighted_csv_to_trimat() {
    // We load moreno_lesmis/out.moreno_lesmis_lesmis. It is in Data directory of the crate.
    println!("\n\n test_weighted_csv_to_trimat");
    log_init_test();
    // path from where we launch cargo test
    let path = Path::new(crate::DATADIR).join("moreno_lesmis").join("out.moreno_lesmis_lesmis");
    log::debug!("\n\n test_weighted_csv_to_trimat, loading file {:?}", path);
    let header_size = get_header_size(&path);
    assert_eq!(header_size.unwrap(),2);
    println!("\n\n test_weighted_csv_to_trimat, data : {:?}", path);
    //
    let trimat_res  = csv_to_trimat::<f64>(&path, false, b' ');
    if let Err(err) = &trimat_res {
        eprintln!("ERROR: {}", err);
        assert_eq!(1,0);
    }
    // now we can dump some info
    let mut trimat_iter = trimat_res.as_ref().unwrap().0.triplet_iter();
    let nodeset = &trimat_res.as_ref().unwrap().1;
    for _ in 0..5 {
        let triplet = trimat_iter.next().unwrap();
        let node1 = nodeset.get_index(triplet.1.0).unwrap();
        let node2 = nodeset.get_index(triplet.1.1).unwrap();
        let value = triplet.0;
        log::debug!("node1 {}, node2 {},  value {} ", node1, node2, value);
    }
} // end test test_weighted_csv_to_trimat



}  // edn of mod tests
