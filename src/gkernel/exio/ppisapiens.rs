//! To load homo_sapiens.mat ppi data
//! It is graph where nodes have multiple labels.
//! The preprocessed file is retrieved from <http://snap.stanford.edu/node2vec> and converted from matlab with file ppi.jl
//! 
//! 

use anyhow::{anyhow};

use std::io::{BufReader};
use std::fs::{OpenOptions};

use csv::ReaderBuilder;
use std::collections::HashMap;
use indexmap::IndexMap;

use petgraph::graph::{Graph, NodeIndex};
use petgraph::stable_graph::{DefaultIx};


use crate::gkernel::pgraph::*;
use crate::gkernel::idmap::*;


// TODO in fact it is same as EuNode, do we provide basic Node of our own?
/// A node is a num as we want to keep track of origin id and a label consisting in a vector of length 1 as we have only one label
pub struct PpiNode {
    /// num of node as given in original data file.
    num : u32,
    /// associated labels. In fact in this example only one label : department num in Eu administration
    labels : Nweight<u8>
}

impl PpiNode {
    pub fn new(num : u32, labels : Nweight<u8>) -> Self {
        PpiNode{ num, labels}
    }

    /// retrieve original node num (given in datafile)
#[allow(unused)]
    pub fn get_num(&self) -> u32 { self.num}


} // end of impl PpiNode


impl HasNweight<u8> for PpiNode {

    fn get_nweight(&self) -> &Nweight<u8> {
        &self.labels
    } // end of get_nweight

} // end of impl HasNweight


/// Edge satisfy default 
#[derive(Default)]
pub struct PpiEdge {
    labels : Eweight<u8>    
}


impl HasEweight<u8> for PpiEdge {
    fn get_eweight(&self) -> &Eweight<u8> {
        &self.labels
    }
} // end of impl HasEweight

//==================================================================================================


/// argument is directory where the 2 files from this example are stored
/// read 2 files : ppi-network.csv for network. it is a csv with blank as separator
/// ppi-labels.csv  for labels. a list of lines : rank node , label each node can occur in many lines.
// The graph is directed. 
#[allow(unused)]
pub fn read_ppi_data(dir : String) -> anyhow::Result< (Graph<PpiNode , PpiEdge , petgraph::Directed, DefaultIx>, IdMap<u8,u8>)> {
    //
    let delim = b' ';
    let nb_fields = 2;
    //
    // first we read labels
    //
    let filepath = std::path::Path::new(&dir).join("ppi-labels.csv");
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("read_ppi_data : reload could not open file {:?}", filepath.as_os_str());
        println!("directed_from_csv could not open file {:?}", filepath.as_os_str());
        return Err(anyhow!("directed_from_csv could not open file {}", filepath.display()));            
    }
    let file = fileres?;
    let bufreader = BufReader::new(file);
    let mut nb_record = 0;
    let mut node : usize = 0;
    let mut label : u8 = 0;
    let mut max_label = 0u8;
    let mut rdr = ReaderBuilder::new().delimiter(delim).flexible(false).has_headers(false).from_reader(bufreader);
    // we have less than 5000 nodes
    let mut nodelabels = HashMap::<usize, Vec::<u8>>::with_capacity(5000);

    for result in rdr.records() {
        let record = result?;
        if record.len() != nb_fields {
            println!("non constant number of fields at record {} first record has {}", nb_record+1, nb_fields);
            return Err(anyhow!("non constant number of fields at record {} first record has {}",nb_record+1, nb_fields));   
        }
        let field = record.get(0).unwrap();
        if let Ok(idx) = field.parse::<usize>() {
            node = idx;
            assert!(node <= 5000);
        }
        else {
            return Err(anyhow!("error decoding field 1 of record  {}",nb_record+1)); 
        }
        // decode field 1
        let field = record.get(1).unwrap();
        if let Ok(lbl) = field.parse::<u8>() {
            label = lbl;
            max_label = max_label.max(label);
        }
        else {
            return Err(anyhow!("error decoding field 2 of record  {}",nb_record+1)); 
        }
        // first time we see node?
        match nodelabels.get_mut(&node) {
            None => {
                let mut labels = Vec::<u8>::new();
                labels.push(label);
                nodelabels.insert(node, labels);
            }
            Some(labels) => {
                labels.push(label);
            }
        }
    } // end for storing labels

    // relabelling scheme if necessary (change 0 to max_label +1 after check! )

    // read network file

    return Err(anyhow!("not yet"));
} // end of read_ppi_data