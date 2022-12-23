//! To load homo_sapiens.mat ppi data
//! It is graph where nodes have multiple labels.
//! 
//! The preprocessed file is retrieved from <http://snap.stanford.edu/node2vec> and converted from matlab with file ppi.jl
//! 
//! The graph is symetric but is given in data file all edges described so can be loaded in both mode Directed, Undirected.

//! 

use anyhow::{anyhow};

use std::io::{BufReader};
use std::fs::{OpenOptions};

use csv::ReaderBuilder;
use std::collections::HashMap;
use indexmap::IndexMap;

use petgraph::graph::{Graph, NodeIndex};
use petgraph::stable_graph::{DefaultIx};


use crate::embed::gkernel::{pgraph::*, idmap::*};


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


/// load ppi labels and returns relabelled scheme.
/// Returns a HashMap giving for each node a vector of labels as read in file  and a remapping scheme for labels
pub fn load_labels(dir : &String) ->  anyhow::Result<(HashMap::<u32, Vec::<u8>>, IndexMap::<u8, u8>) > {
    let delim = b' ';
    let nb_fields = 2;
    //
    // first we read labels
    //
    let filepath = std::path::Path::new(&dir).join("ppi-labels.csv");
    log::info!("\n reading label file : {:?}", filepath);
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("read_ppi_data : reload could not open file {:?}", filepath.as_os_str());
        println!("directed_from_csv could not open file {:?}", filepath.as_os_str());
        return Err(anyhow!("directed_from_csv could not open file {}", filepath.display()));            
    }
    let file = fileres?;
    let bufreader = BufReader::new(file);
    let mut nb_record = 0;
    let mut node : u32;
    let mut label : u8;
    let mut max_label = 0u8;
    let mut rdr = ReaderBuilder::new().delimiter(delim).flexible(false).has_headers(false).from_reader(bufreader);
    // we have less than 5000 nodes
    let mut nodelabels = HashMap::<u32, Vec::<u8>>::with_capacity(5000);

    for result in rdr.records() {
        let record = result?;
        if record.len() != nb_fields {
            log::error!("record num : {}, record : {:?}, record length : {}", nb_record, &record,  record.len());
            println!("non constant number of fields at record {} first record has {}", nb_record+1, nb_fields);
            return Err(anyhow!("non constant number of fields at record {} first record has {}",nb_record+1, nb_fields));   
        }
        let field = record.get(0).unwrap();
        if let Ok(idx) = field.parse::<u32>() {
            node = idx;
            assert!(node <= 5000);
        }
        else {
            log::info!("error decoding field 1 of record  {}",nb_record+1);
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
        nb_record = nb_record + 1; 
    } // end for storing labels
    log::info!("end reading labels");
    // relabelling scheme if necessary (change 0 to max_label +1 after check! )
    assert!(max_label < u8::MAX);
    let mut relabel =  IndexMap::<u8, u8>::new();
    let val_for_0 = max_label+1;
    for v in nodelabels.values_mut() {
        for l in v {
            if *l == 0 {
                if relabel.get(l).is_none() {
                    // first time we see 0
                    relabel.insert(*l, val_for_0);
                }
                *l = val_for_0;
            }
            else {
                if relabel.get(l).is_none() {
                    // first time we see l != 0
                    relabel.insert(*l, *l);
                }                
            }
        }
    }
    log::info!("end relabelling labels");
    //
    return Ok((nodelabels,relabel));
} // end of load_labels



/// argument is directory where the 2 files from this example are stored
/// read 2 files : ppi-network.csv for network. it is a csv with blank as separator
/// ppi-labels.csv  for labels. a list of lines : rank node , label each node can occur in many lines.
// The graph is directed. 
#[allow(unused)]
pub fn read_ppi_directed_data(dir : String) -> anyhow::Result< (Graph<PpiNode , PpiEdge , petgraph::Directed, DefaultIx>, IdMap<u8,u8>)> {
    //
    let delim = b' ';
    let relabels = load_labels(&dir);
    if relabels.is_err() {
        log::error!("read_ppi_directed_data could, call to ppisapiens::load_labels failed");
    }
    let (nodelabels, relabel) = relabels.unwrap();
    //
    // read network file
    //
    let filepath = std::path::Path::new(&dir).join("ppi-network.csv");
    log::info!("reading network file : {:?}", filepath);
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("read_ppi_data : reload could not open file {:?}", filepath.as_os_str());
        println!("directed_from_csv could not open file {:?}", filepath.as_os_str());
        return Err(anyhow!("directed_from_csv could not open file {}", filepath.display()));            
    }
    let file = fileres?;
    let bufreader = BufReader::new(file);
    let nb_fields = 3;
    let mut nb_record = 0;
    let mut nb_self_loops = 0;
    let mut nb_double_dir = 0; // to count how many are given twice (data file give non diag edge twice)
    let mut node1 : u32;
    let mut node2 : u32;
    let mut gnode1 : Option<NodeIndex>;
    let mut gnode2 : Option<NodeIndex>;
    //
    let mut graph = Graph::<PpiNode, PpiEdge, petgraph::Directed>::new();
    // This is to retrieve NodeIndex given original num of node as given in data file
    let mut nodeset = IndexMap::<u32, NodeIndex>::new();
    //
    let mut rdr = ReaderBuilder::new().delimiter(delim).flexible(false).has_headers(true).from_reader(bufreader);
    log::info!("reading records");
    for result in rdr.records() {
        log::debug!("edge record : {}", nb_record);
        let record = result?;
        if record.len() != nb_fields {
            log::error!("record num : {}, record : {:?}, record length : {}", nb_record, &record,  record.len());
            println!("non constant number of fields at record {} first record has {}", nb_record+1, nb_fields);
            return Err(anyhow!("non constant number of fields at record {} first record has {}",nb_record+1, nb_fields));   
        }
        // decode node1
        let field = record.get(0).unwrap();
        if let Ok(idx) = field.parse::<u32>() {
            node1 = idx;
            assert!(node1 <= 5000);
            if !nodeset.contains_key(&node1) {
                // we construct node for Graph, recall that nodelabels are already relabelled 
                let labels = nodelabels.get(&node1).unwrap();
                let labels = Nweight::<u8>::new(labels.clone());
                // graph.add_node
                gnode1 = Some(graph.add_node(PpiNode::new(node1, labels)));
                nodeset.insert(node1, gnode1.unwrap());
            }
            else { // node already in graph, we must retrieve its index
                gnode1 = Some(*nodeset.get(&node1).unwrap());
            }
        }
        else {
            log::error!("error decoding node 2 of record  {}",nb_record+1);
            return Err(anyhow!("error decoding field 1 of record  {}",nb_record+1)); 
        }
        // decode node2 
        let field = record.get(1).unwrap();
        if let Ok(idx) = field.parse::<u32>() {
            node2 = idx;
            if !nodeset.contains_key(&node2) {
                // we construct node for Graph, recall that nodelabels are already relabelled 
                let labels = nodelabels.get(&node2).unwrap();
                let labels = Nweight::<u8>::new(labels.clone());
                // graph.add_node
                gnode2 = Some(graph.add_node(PpiNode::new(node2, labels)));
                nodeset.insert(node2, gnode2.unwrap());
            }
            else { // node already in graph, we must retrieve its index
                gnode2 = Some(*nodeset.get(&node2).unwrap());
            }            
        }
        else {
            log::error!("error decoding node 2 of record  {}",nb_record+1);
            return Err(anyhow!("error decoding node 2 of record  {}",nb_record+1)); 
        }
        // graph is directed. count number of double dir edges
        if graph.contains_edge(gnode2.unwrap(), gnode1.unwrap()) {
            nb_double_dir += 1;
        }
        graph.update_edge(gnode1.unwrap(), gnode2.unwrap(), PpiEdge::default());
        if gnode1.unwrap() == gnode2.unwrap() {
            nb_self_loops += 1;
        }
        nb_record += 1;
    }  // end of edge parsing
    //
    log::info!("nb nodes = {}", graph.raw_nodes().len());
    log::info!("nb edges = {}", graph.raw_edges().len());
    //
    let idmap = IdMap::<u8,u8>::new(nodeset, relabel);
    //
    Ok((graph,idmap))
} // end of read_ppi_data



#[allow(unused)]
pub fn read_ppi_undirected_data(dir : String) -> anyhow::Result< (Graph<PpiNode , PpiEdge , petgraph::Undirected, DefaultIx>, IdMap<u8,u8>)> {
    //
    let delim = b' ';
    let relabels = load_labels(&dir);
    if relabels.is_err() {
        log::error!("read_ppi_undirected_data could, call to ppisapiens::load_labels failed");
    }
    let (nodelabels, relabel) = relabels.unwrap();
    //
    // read network file as a symetric (Undirected) graph
    //
    let filepath = std::path::Path::new(&dir).join("ppi-network.csv");
    log::info!("reading network file : {:?}", filepath);
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("read_ppi_data : reload could not open file {:?}", filepath.as_os_str());
        println!("directed_from_csv could not open file {:?}", filepath.as_os_str());
        return Err(anyhow!("directed_from_csv could not open file {}", filepath.display()));            
    }
    let file = fileres?;
    let bufreader = BufReader::new(file);
    let nb_fields = 3;
    let mut nb_record = 0;
    let mut node1 : u32;
    let mut node2 : u32;
    let mut nb_self_loops = 0;
    let mut nb_double_dir = 0; // to count how many are given twice (data file give non diag edge twice)
    let mut gnode1 : Option<NodeIndex>;
    let mut gnode2 : Option<NodeIndex>;
    //
    let mut graph = Graph::<PpiNode, PpiEdge, petgraph::Undirected>::new_undirected();
    // This is to retrieve NodeIndex given original num of node as given in data file
    let mut nodeset = IndexMap::<u32, NodeIndex>::new();
    //
    let mut rdr = ReaderBuilder::new().delimiter(delim).flexible(false).has_headers(true).from_reader(bufreader);
    log::info!("reading records");
    for result in rdr.records() {
        log::debug!("edge record : {}", nb_record);
        let record = result?;
        if record.len() != nb_fields {
            log::error!("record num : {}, record : {:?}, record length : {}", nb_record, &record,  record.len());
            println!("non constant number of fields at record {} first record has {}", nb_record+1, nb_fields);
            return Err(anyhow!("non constant number of fields at record {} first record has {}",nb_record+1, nb_fields));   
        }
        // decode node1
        let field = record.get(0).unwrap();
        if let Ok(idx) = field.parse::<u32>() {
            node1 = idx;
            assert!(node1 <= 5000);
            if !nodeset.contains_key(&node1) {
                // we construct node for Graph, recall that nodelabels are already relabelled 
                let labels = nodelabels.get(&node1).unwrap();
                let labels = Nweight::<u8>::new(labels.clone());
                // graph.add_node
                gnode1 = Some(graph.add_node(PpiNode::new(node1, labels)));
                nodeset.insert(node1, gnode1.unwrap());
            }
            else { // node already in graph, we must retrieve its index
                gnode1 = Some(*nodeset.get(&node1).unwrap());
            }
        }
        else {
            log::error!("error decoding node 2 of record  {}",nb_record+1);
            return Err(anyhow!("error decoding field 1 of record  {}",nb_record+1)); 
        }
        // decode node2 
        let field = record.get(1).unwrap();
        if let Ok(idx) = field.parse::<u32>() {
            node2 = idx;
            if !nodeset.contains_key(&node2) {
                // we construct node for Graph, recall that nodelabels are already relabelled 
                let labels = nodelabels.get(&node2).unwrap();
                let labels = Nweight::<u8>::new(labels.clone());
                // graph.add_node
                gnode2 = Some(graph.add_node(PpiNode::new(node2, labels)));
                nodeset.insert(node2, gnode2.unwrap());
            }
            else { // node already in graph, we must retrieve its index
                gnode2 = Some(*nodeset.get(&node2).unwrap());
            }            
        }
        else {
            log::error!("error decoding node 2 of record  {}",nb_record+1);
            return Err(anyhow!("error decoding node 2 of record  {}",nb_record+1)); 
        }
        // graph is directed. count number of double dir edges
        if graph.contains_edge(gnode2.unwrap(), gnode1.unwrap()) {
            nb_double_dir += 1;
        }
        else {
            // we load as a symetric graph, we update edge if we did not already see reverse edge
            graph.update_edge(gnode1.unwrap(), gnode2.unwrap(), PpiEdge::default());
        }
        if gnode1.unwrap() == gnode2.unwrap() {
            nb_self_loops += 1;
        }
        nb_record += 1;
    }  // end of edge parsing
    log::info!("nb self loops : {}, nb bidirectional edges {}", nb_self_loops, nb_double_dir);
    log::info!("nb nodes = {}", graph.raw_nodes().len());
    log::info!("nb edges = {}", graph.raw_edges().len());
    //
    let idmap = IdMap::<u8,u8>::new(nodeset, relabel);
    //
    Ok((graph,idmap))
} // end of read_ppi_undirected_data

//=====================================================================================


#[cfg(test)]
mod tests {


const PPI_DIR:&str = "/home/jpboth/Data/Graphs/PPI";

use super::*; 


fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}


#[test]
fn test_load_ppi_directed() {
    log_init_test();
    //
    //
    let res_graph = read_ppi_directed_data(String::from(PPI_DIR));
    assert!(res_graph.is_ok());
    //
    let (graph, _nodeset) = res_graph.unwrap();
    //
    log::info!("ppi nodes : {}, ppi edge : {}", graph.raw_nodes().len(), graph.raw_edges().len());
} // end of test_load_ppi_directed


#[test]
fn test_load_ppi_undirected() {
    log_init_test();
    //
    //
    let res_graph = read_ppi_undirected_data(String::from(PPI_DIR));
    assert!(res_graph.is_ok());
    //
    let (graph, _nodeset) = res_graph.unwrap();
    //
    log::info!("ppi nodes : {}, ppi edge : {}", graph.raw_nodes().len(), graph.raw_edges().len());
} // end of test_load_ppi_undirected

}  // end of mod tests