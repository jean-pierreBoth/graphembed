//! Construct or dump a (small) graph or its matricial representation FULL or CSR from data from a csv file


//use std::fmt::{Debug};

use log::*;
use anyhow::{anyhow};

use std::collections::HashSet;

use std::fs::{OpenOptions};
use std::path::{Path};
use std::str::FromStr;

use std::io::{Read};

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

/// maps the type N id of a node to a rank in a matrix
pub type NodeIndexation<N> = IndexSet<N>;


// count number of first lines beginning with '#' or '%'
fn get_header_size(filepath : &Path) -> anyhow::Result<usize> {
    //
    log::info!("get_header_size");
    //
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("ProcessingState reload_json : reload could not open file {:?}", filepath.as_os_str());
        println!("directed_from_csv could not open file {:?}", filepath.as_os_str());
        return Err(anyhow!("directed_from_csv could not open file {}", filepath.display()));            
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
            log::debug!("file has {} nb headrers lines", nb_header_lines);
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
/// Returns the MatRepr field and a mapping from NodeId to a rank in matrix.
pub fn csv_to_csrmat<F:Float+FromStr>(filepath : &Path, directed : bool, delim : u8) -> anyhow::Result<(MatRepr<F>, NodeIndexation<usize>)> 
    where F: FromStr + Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + Default {
    let res_csv = csv_to_trimat(filepath, directed, delim);
    if res_csv.is_ok() {
        let csrmat : CsMat<F> = res_csv.as_ref().unwrap().0.to_csr();
        return Ok((MatRepr::from_csrmat(csrmat), res_csv.unwrap().1));
    }
    else {
        return Err(res_csv.unwrap_err());
    }
}  // end of csv_to_csrmat


/// load a directed/undirected  weighted/unweighted graph in csv format into a TriMatI representation.  
/// delim is the delimiter used in the csv file necessary for csv::ReaderBuilder
/// If there are 3 fields by record, the third is assumed to be a weight convertible type F (morally usize, f32 or f64)
/// nodes must be numbered contiguously from 0 to nb_nodes-1 to be stored in a matrix.
/// Returns a 2-uple containing first the TriMatI and then the NodeIndexation remapping nodes into (0..nb_nodes) 
/// 
pub fn csv_to_trimat<F:Float+FromStr>(filepath : &Path, directed : bool, delim : u8) -> anyhow::Result<(TriMatI<F, usize>, NodeIndexation<usize>)> 
    where F: FromStr + Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + Default {
    //
    // first get number of header lines
    let nb_headers_line = get_header_size(&filepath)?;
    log::info!("directed_from_csv , got header nb lines {}", nb_headers_line);
    //
    // as nodes num in csv files are guaranteed to be numbered contiguously in 0..nb_nodes
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
    let nb_edges_guess = 500_000;   // to pass as function argument
    let mut rows = Vec::<usize>::with_capacity(nb_edges_guess);
    let mut cols = Vec::<usize>::with_capacity(nb_edges_guess);
    let mut values = Vec::<F>::with_capacity(nb_edges_guess);
    let mut node1 : usize;
    let mut node2 : usize;
    let mut weight : F;
    let mut rowmax : usize = 0;
    let mut colmax : usize = 0;
    let mut nb_record = 0;
    let mut nb_fields = 0;
    let mut nb_nodes : usize = 0;
    //
    // nodes must be numbered contiguously from 0 to nb_nodes-1 to be stored in a matrix.
    let mut rdr = ReaderBuilder::new().delimiter(delim).flexible(false).has_headers(false).from_reader(file);
    for result in rdr.records() {
        let record = result?;
        if log::log_enabled!(Level::Info) && nb_record <= 5 {
            log::info!(" record num {:?}, {:?}", nb_record, record);
        }
        //
        if nb_record == 0 {
            nb_fields = record.len();
            log::info!("nb fields = {}", nb_fields);
        }
        else {
            if record.len() != nb_fields {
                println!("non constant number of fields at record {} first record has {}",nb_record+1,  nb_fields);
                return Err(anyhow!("non constant number of fields at record {} first record has {}",nb_record+1,  nb_fields));   
            }
        }
        // we have 2 or 3 fields
        let field = record.get(0).unwrap();
        // decode into Ix type
        if let Ok(node) = field.parse::<usize>() {
            let already = nodeindex.get_index_of(&node);
            match already {
                Some(idx) => { node1 = idx},
                None             => {   node1 = nodeindex.insert_full(node).0;
                                        log::debug!("inserting node num : {}, rank : {}", node, node1);
                                        nb_nodes += 1;
                                    }
            }
            rowmax = rowmax.max(node1);
        }
        else {
            return Err(anyhow!("error decoding field 1 of record  {}",nb_record+1)); 
        }
        let field = record.get(1).unwrap();
        if let Ok(node) = field.parse::<usize>() {
            let already = nodeindex.get_index_of(&node);
            match already {
                Some(idx) => { node2 = idx},
                None             => {   node2 = nodeindex.insert_full(node).0;
                                        log::debug!("inserting node num : {}, rank : {}", node, node2);
                                        nb_nodes += 1;
                                    }
            }            
            colmax = colmax.max(node2);
        }
        else {
            return Err(anyhow!("error decoding field 2 of record  {}",nb_record+1)); 
        }
        if !hset.insert((node1,node2)) {
            log::error!("2-uple ({:?}, {:?}) already present, record {}", node1, node2, nb_record);
            return Err(anyhow!("2-uple ({:?}, {:?}) already present", node1, node2));
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
                return Err(anyhow!("error decoding field 2 of record  {}",nb_record+1)); 
            }
        }
        else {
            weight = F::one();
        }
        // we store data ...
        rows.push(node1);
        cols.push(node2);
        values.push(weight);
        if !directed {
            // store symetric point and check it was not already present
            if !hset.insert((node2,node1)) {
                log::error!("undirected case 2-uple ({:?}, {:?}) already present, record {}", node2, node1, nb_record);
                return Err(anyhow!("2-uple ({:?}, {:?}) already present", node2, node1));
            }
            rows.push(node2);
            cols.push(node1);
            values.push(weight); 
        }
        nb_record += 1;
        if log::log_enabled!(Level::Info) && nb_record <= 5 {
            log::info!("{:?}", record);
            log::info!(" node1 {:?}, node2 {:?}", node1, node2);
        }
    }  // end of for result
    //
    assert_eq!(rows.len(), cols.len());
    log::info!("csv file read!, nb_record {}", nb_record);
    log::info!("rowmax : {}, colmax : {}", rowmax, colmax);
    let trimat = TriMatI::<F, usize>::from_triplets((rows.len(), rows.len()), rows, cols, values);
    //
    assert_eq!(nb_nodes, nodeindex.len());
    //
    Ok((trimat, nodeindex))
} // end of csv_to_trimat


//========================================================================================

#[cfg(test)]

mod tests {

//    cargo test csv  -- --nocapture
//    cargo test csv::tests::test_name -- --nocapture
//    RUST_LOG=graphembed::io::csv=TRACE cargo test csv -- --nocapture


const DATADIR : &str = &"/home/jpboth/Rust/graphembed/Data";

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
    // We load CA-GrQc.txt taken from Snap data directory. It is in Data directory of the crate.
    log_init_test();
    // path from where we launch cargo test
    let path = Path::new(DATADIR).join("ca-GrQc.txt");
    //
    let header_size = get_header_size(&path);
    assert_eq!(header_size.unwrap(),4);
    println!("\n\n test_directed_unweighted_csv_to_graph data : {:?}", path);
    // we must have 28979 edges as we have 28979 records.
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
    let path = Path::new(DATADIR).join("moreno_lesmis").join("out.moreno_lesmis_lesmis");
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
