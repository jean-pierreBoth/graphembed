//! Construct or dump a (small) graph or its matricial representation FULL or CSR from data from a csv file


//use std::fmt::{Debug};

use log::*;
use anyhow::{anyhow};


use std::fs::{OpenOptions};
use std::path::{Path};
use std::str::FromStr;

use std::io::{Read};

use csv::ReaderBuilder;
use num_traits::float::*;

use annembed::tools::svdapprox::*;

//use petgraph::graph::{Graph, NodeIndex, IndexType};
use petgraph::graphmap::{GraphMap, NodeTrait};
#[allow(unused)]
use petgraph::{Directed,EdgeType};



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




/// load a directed unweighted graph in csv format into a MatRepr representation
/// nodes must be numbered contiguously from 0 to nb_nodes-1 to be stored in a matrix
/// until we use an IndexSet (see annembed::fromhnsw::KGraph)
pub fn directed_unweighted_csv_to_csrmat<N, F:Float>(filepath : &Path, delim : u8, mat_type :  MatType) -> anyhow::Result<MatRepr<F>> 
        where   N : NodeTrait + std::hash::Hash + std::cmp::Eq + FromStr + std::fmt::Display {
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
    let nb_edges_guess = 500_000;   // to pass as function argument
    let mut rows = Vec::<usize>::with_capacity(nb_edges_guess);
    let mut cols = Vec::<usize>::with_capacity(nb_edges_guess);
    let mut values = Vec::<f64>::with_capacity(nb_edges_guess);
    let mut node1 : N;
    let mut node2 : N;
    // now we can parse records and construct a parser from current position of file
    // we already skipped headers
    let mut nb_record = 0;
    let mut nb_fields = 0;
    //
    let mut rdr = ReaderBuilder::new().delimiter(delim).flexible(false).has_headers(false).from_reader(file);
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
        }
        else {
            return Err(anyhow!("error decoding field 1 of record  {}",nb_record+1)); 
        }
        let field = record.get(1).unwrap();
        if let Ok(idx) = field.parse::<N>() {
            node2 = idx;
        }
        else {
            return Err(anyhow!("error decoding field 2 of record  {}",nb_record+1)); 
        }
        // TODO store data ...
        nb_record += 1;
        if log::log_enabled!(Level::Info) && nb_record <= 5 {
            log::info!("{:?}", record);
            log::info!(" node1 {}, node2 {}", node1, node2);
        }
        //
        
    }  // end of for result
    //
    Err(anyhow!("directed_unweighted_csv_to_mat unimplemented"))
}  // end of directed_unweighted_csv_to_mat



//========================================================================================

#[cfg(test)]

mod tests {

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
    println!("\n\n test_directed_unweighted_from_csv data : {:?}", path);
    // we must have 28979 edges as we have 28979 records.
    let graph = directed_unweighted_csv_to_graph::<u32, Directed>(&path, b'\t');
    if let Err(err) = &graph {
        eprintln!("ERROR: {}", err);
    }
    assert!(graph.is_ok());

} // end test test_directed_unweightedfrom_csv




#[test]
fn test_directed_weighted_from_csv() {
    // We load moreno_lesmis/out.moreno_lesmis_lesmis. It is in Data directory of the crate.
    println!("\n\n test_undirected_weighted_from_csv");
    log_init_test();

} // end test test_directed_unweightedfrom_csv



}  // edn of mod tests
