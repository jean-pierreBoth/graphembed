/// Provides a function to load Mail-Eu labeled graph <https://snap.stanford.edu/data/email-Eu-core.html>
/// 1005 nodes , 25571 edges , labels between 0 and 41 

use anyhow::{anyhow};

use std::io::{BufReader};
use std::fs::{OpenOptions};

use csv::ReaderBuilder;
use indexmap::IndexMap;

use petgraph::graph::{Graph, NodeIndex};
use petgraph::stable_graph::{DefaultIx};


use crate::gkernel::pgraph::*;
use crate::gkernel::idmap::*;


/// A node is a num as we want to keep track of origin id and a label consisting in a vector of length 1 as we have only one label
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
#[allow(unused)]
    pub fn get_num(&self) -> u32 { self.num}


} // end of impl EuNode


impl HasNweight<u8> for EuNode {

    fn get_nweight(&self) -> &Nweight<u8> {
        &self.labels
    } // end of get_nweight

} // end of impl HasNweight


/// Edge satisfy default 
#[derive(Default)]
pub struct EuEdge {
    labels : Eweight<u8>    
}


impl HasEweight<u8> for EuEdge {
    fn get_eweight(&self) -> &Eweight<u8> {
        &self.labels
    }
} // end of impl HasEweight

//==================================================================================================

/// argument is directory where the 2 files from this example are stored
/// read 2 files : email-Eu-core.txt for network. it is a csv with blank as separator
/// email-Eu-core-department-labels.txt for labels. a list of lines : rank node , label
// The graph is directed. 
#[allow(unused)]
pub fn read_maileu_data(dir : String) -> anyhow::Result< (Graph<EuNode , EuEdge , petgraph::Directed, DefaultIx>, IdMap<u8,u8>)> {
    //
    let delim = b' ';
    let nb_fields = 2;
    //
    // first we read labels
    //
    let filepath = std::path::Path::new(&dir).join("email-Eu-core-department-labels.txt");
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("read_maileu_data : reload could not open file {:?}", filepath.as_os_str());
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
    // now we run relabelling scheme. We know max label is 41, we just remap label 0 to 42 and leave others unchanged
    // 
    let mut relabel =  IndexMap::<u8, u8>::new();
    for i in &nodelabels {
        if *i == 0 {
            relabel.insert(*i, 42u8);
        }
        else {
            relabel.insert(*i, *i);
        }
    }
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
    let mut nb_self_loops = 0;
    let mut nb_double_dir = 0; // to count how many are given twice (data file give non diag edge twice)
    let mut node1 : u32;
    let mut node2 : u32;
    let mut gnode1 : Option<NodeIndex>;
    let mut gnode2 : Option<NodeIndex>;
    //
    let mut graph = Graph::<EuNode, EuEdge, petgraph::Directed>::new();
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
                let relabelled = relabel.get(&label).unwrap();
                if log::log_enabled!(log::Level::Debug) {
                    if label == 0 {
                        log::info!("found label 0 at node {}", node1);
                        assert_eq!(*relabelled, 42u8);
                    }
                }
                let labels = Nweight::<u8>::new(Vec::from([*relabelled]));
                // graph.add_node
                gnode1 = Some(graph.add_node(EuNode::new(node1, labels)));
                nodeset.insert(node1, gnode1.unwrap());
            }
            else { // node already in graph, we must retrieve its index
                gnode1 = Some(*nodeset.get(&node1).unwrap());
            }
        }
        else {
            return Err(anyhow!("error decoding field 1 of record  {}",nb_record+1)); 
        }
        // read second field
        let field = record.get(1).unwrap();
        if let Ok(idx) = field.parse::<u32>() {
            node2 = idx;
            if !nodeset.contains_key(&node2) {
                // we construct node for Graph
                let label = nodelabels[node2 as usize];
                let relabelled = relabel.get(&label).unwrap();
                let labels = Nweight::<u8>::new(Vec::from([*relabelled]));
                // graph.add_node
                gnode2 = Some(graph.add_node(EuNode::new(node2, labels)));
                nodeset.insert(node2, gnode2.unwrap());
            }
            else { // node already in graph, we must retrieve its index
                gnode2 = Some(*nodeset.get(&node2).unwrap());
            }            
        }
        else {
            return Err(anyhow!("error decoding field 2 of record  {}",nb_record+1)); 
        }
        // graph is directed. count number of double dir edges
        if graph.contains_edge(gnode2.unwrap(), gnode1.unwrap()) {
            nb_double_dir += 1;
        }
        graph.update_edge(gnode1.unwrap(), gnode2.unwrap(), EuEdge::default());
        if gnode1.unwrap() == gnode2.unwrap() {
            nb_self_loops += 1;
        }
        log::debug!(" index1 {} , index2 {}, gnode1 {:?} gnode2 {:?} edge count {}", node1, node2, gnode1.unwrap(), gnode2.unwrap(), graph.edge_count());
        nb_record += 1;
    } // end of nb records
    //
    log::info!("nb self loops : {}, nb bidirectional edges {}", nb_self_loops, nb_double_dir);
    log::info!("nb nodes = {}", graph.raw_nodes().len());
    log::info!("nb edges = {}", graph.raw_edges().len());
    //
    let idmap = IdMap::<u8,u8>::new(nodeset, relabel);
    //
    Ok((graph,idmap))
} // end of read_maileu_data


//=====================================================================================


#[cfg(test)]
mod tests {


const MAILEU_DIR:&str = "/home/jpboth/Data/Graphs/Mail-EU";

use super::*; 


fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}


#[test]
fn test_load_maileu() {
    log_init_test();
    //
    let res_graph = read_maileu_data(String::from(MAILEU_DIR));
    assert!(res_graph.is_ok());
    //
    let (graph, _nodeset) = res_graph.unwrap();
    // check number of nodes and edges.
    assert_eq!(graph.raw_nodes().len(), 1005);
    assert_eq!(graph.raw_edges().len(), 25571);
    // access via NodeIndex
    let first_node_weight = graph.node_weight(NodeIndex::new(0)).unwrap();
    let first_node_nweight = first_node_weight.get_nweight();
    let first_node_labels = first_node_nweight.get_labels();
    assert_eq!(first_node_labels.len(), 1);
    assert_eq!(first_node_labels[0], 1); // We incremented labels by 1!
    // access via raw_nodes
    let first_node_weight = &graph.raw_nodes()[0].weight;
    let first_node_nweight = first_node_weight.get_nweight();
    let first_node_labels = first_node_nweight.get_labels();
    assert_eq!(first_node_labels.len(), 1);
    assert_eq!(first_node_labels[0], 1);
    // node 130 has label 0 in datafile, translated to 42
    let node130_weight = graph.node_weight(NodeIndex::new(130)).unwrap();
    let node130_nweight = node130_weight.get_nweight();
    let node130_nweight_labels = node130_nweight.get_labels();
    assert_eq!(node130_nweight_labels.len(), 1);
    assert_eq!(node130_nweight_labels[0], 42u8);
} // end of test_load_maileu


}  // end of mod tests