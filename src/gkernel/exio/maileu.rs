/// just to load Mail-Eu labeled graph <https://snap.stanford.edu/data/email-Eu-core.html>
/// 1005 nodes , 25571 edges , labels between 0 and 41 

use anyhow::{anyhow};

use std::io::{BufReader};
use std::fs::{OpenOptions};

use csv::ReaderBuilder;

use petgraph::graph::{Graph, NodeIndex};
use petgraph::stable_graph::{DefaultIx};


use crate::gkernel::pgraph::*;

use indexmap::IndexMap;

/// A node is a num a swe want to keep track of origin id and a label consisting in a one dimensional vector
pub struct EuNode {
    /// num of node as given in original data file.
    num : u32,
    /// associated labels. In fact in this example only one label : department num in Eu administration
    labels : Nweight<u8>
}

impl EuNode {
    pub fn new(num : u32, labels : Nweight<u8>) -> Self {
        EuNode{ num, labels}
    }

    /// retrieve original node num (given in datafile)
    pub fn getnum(&self) -> u32 { self.num}

    /// retrieve associated labels
    pub fn getlabels(&self) -> &Nweight<u8> { &self.labels}

} // end of impl EuNode


impl HasNweight<u8> for EuNode {

    fn get_nweight(&self) -> &Nweight<u8> {
        &self.labels
    } // end of get_nweight

} // end of impl HasNweight


pub struct EuEdge {}

//==================================================================================================

/// argumnt is directory where the 2 files from this example are stored
/// read 2 files : email-Eu-core.txt for network. it is a csv with blank as separator
/// email-Eu-core-department-labels.txt for labels. a list of lines : rank node , label
// The graph is symetric. 
#[allow(unused)]
fn read_maileu_data(dir : String) -> Result< Graph<EuNode , EuEdge, petgraph::Undirected, DefaultIx> , anyhow::Error> {
    //
    let delim = b' ';
    let nb_fields = 2;
    //
    // first we read labels
    //
    let filepath = std::path::Path::new(&dir).join("email-Eu-core-department-labels.txt");
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("ProcessingState reload_json : reload could not open file {:?}", filepath.as_os_str());
        println!("directed_from_csv could not open file {:?}", filepath.as_os_str());
        return Err(anyhow!("directed_from_csv could not open file {}", filepath.display()));            
    }
    let file = fileres?;
    let bufreader = BufReader::new(file);
    //
    let mut nb_record = 0;
    let mut node : usize;
    let mut label : u8;
    let mut rdr = ReaderBuilder::new().delimiter(delim).flexible(false).has_headers(false).from_reader(bufreader);
    let mut nodelabels = Vec::<u8>::with_capacity(1005);
    //
    for result in rdr.records() {
        let record = result?;
        if record.len() != nb_fields {
            println!("non constant number of fields at record {} first record has {}", nb_record+1, nb_fields);
            return Err(anyhow!("non constant number of fields at record {} first record has {}",nb_record+1, nb_fields));   
        }
        let field = record.get(0).unwrap();
        if let Ok(idx) = field.parse::<usize>() {
            node = idx;
            assert!(node <= 1005);
        }
        else {
            return Err(anyhow!("error decoding field 1 of record  {}",nb_record+1)); 
        }
        let field = record.get(1).unwrap();
        if let Ok(lbl) = field.parse::<u8>() {
            label = lbl;
        }
        else {
            return Err(anyhow!("error decoding field 2 of record  {}",nb_record+1)); 
        }
        nodelabels.push(label);
        nb_record += 1;
        //
    } // end of reading records
    // 
    // now we read graph as a csv file
    //
    let filepath = std::path::Path::new(&dir).join("email-Eu-core.txt");
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("ProcessingState reload_json : reload could not open file {:?}", filepath.as_os_str());
        println!("directed_from_csv could not open file {:?}", filepath.as_os_str());
        return Err(anyhow!("directed_from_csv could not open file {}", filepath.display()));            
    }
    let file = fileres?;
    let bufreader = BufReader::new(file);
    nb_record = 0;
    let mut node1 : u32;
    let mut node2 : u32;
    let mut gnode1 : Option<NodeIndex> = None;
    let mut gnode2 : Option<NodeIndex> = None;
    //
    let mut graph = Graph::<EuNode, EuEdge, petgraph::Undirected>::new_undirected();
    // This is to retrieve NodeIndex given original num of node as given in data file
    let mut nodeset = IndexMap::<u32, NodeIndex>::new();
    //
    let mut rdr = ReaderBuilder::new().delimiter(delim).flexible(false).has_headers(false).from_reader(bufreader);
    for result in rdr.records() {
        let record = result?;
        if record.len() != nb_fields {
            println!("non constant number of fields at record {} first record has {}", nb_record+1, nb_fields);
            return Err(anyhow!("non constant number of fields at record {} first record has {}",nb_record+1, nb_fields));   
        }
        let field = record.get(0).unwrap();
        if let Ok(idx) = field.parse::<u32>() {
            node1 = idx;
            if !nodeset.contains_key(&node1) {
                // we construct node for Graph
                let label = nodelabels[node1 as usize];
                let labels = Nweight::<u8>::new(Vec::from([label]));
                // graph.add_node
                gnode1 = Some(graph.add_node(EuNode::new(node1, labels)));
                nodeset.insert(node1, gnode1.unwrap());
            }
        }
        else {
            return Err(anyhow!("error decoding field 1 of record  {}",nb_record+1)); 
        }
        let field = record.get(1).unwrap();
        if let Ok(idx) = field.parse::<u32>() {
            node2 = idx;
            if !nodeset.contains_key(&node2) {
                // we construct node for Graph
                let label = nodelabels[node2 as usize];
                let labels = Nweight::<u8>::new(Vec::from([label]));
                // graph.add_node
                gnode2 = Some(graph.add_node(EuNode::new(node2, labels)));
                nodeset.insert(node2, gnode2.unwrap());
            }
        }
        else {
            return Err(anyhow!("error decoding field 2 of record  {}",nb_record+1)); 
        }
        // we can add the  edge once! and only once, so we must check if it is already stored
        graph.update_edge(gnode1.unwrap(), gnode2.unwrap(), EuEdge{});
    }
    //
    Ok(graph)
} // end of read_maileu_data