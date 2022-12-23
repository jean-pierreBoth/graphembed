//! Graph can come with many types of discrete labels. We need to map original labels of graph data files to discrete labels.
//! 
//! It is possible to have labels in datafile that can be mapped to our discrete labels but that 
//! are neither strings nor u6, u16 etc.  
//! Moreover the default initialization of labels (i.E 0 for u8, u16 etc)
//! is used internally to cover the case where a  node has no input (or output) edge. In this case
//! the embedded vector is represented by the default value (i.e 0). 
//! **So effective labels stored in the graph must not be 0!!**
//! 
//! We must also store reordering of nodes.
//! 

use std::hash::Hash;

use petgraph::stable_graph::{NodeIndex};

use indexmap::IndexMap;

use super::pgraph::*;


// TODO Do we keep u32 or do we parametrize on initial node Id?

/// ToLabel is the original label of nodes (or edge) of graph, Label is discrete label associated to Nodes in MgraphSketch
pub struct IdMap<ToLabel, Label> 
    where Label   : LabelT ,
          ToLabel : Eq + Hash {
    /// given a node rank in file get the NodeIndex in graph
    ranktoidx : IndexMap::<u32, NodeIndex>,
    /// given the original label, get the discrete label used in node labeling
    relabel : IndexMap<ToLabel, Label>
} // end of struct IdMap


/// When loading a datafile, A graph is returned together with a structure satisfying trait IdMapper
/// 
/// When a node has no input edge or no output edge, if we want to get an Array2 are embedding result we must define a specific label
/// to fill the corresponding In or Out sketching vector. This is the Default label! 
/// (An alternative woud have been to return an Option\<Vec\> for each node, but still we would have to relabel to fit into our labels)
/// Moreover we may need to map labels in datafile to out implemented labels. So there is a specific relabelling pass in reading data
/// Typically the default label 0 must be remapped to someting else (max label used + 1) 

/// In the same way at the end of the embedding process we do have correspondance between embedded vectors (accessed by a row in embbed Array2 corresponding
/// to a NodeIndex) and initial rank of node
/// 

pub trait IdMapper<ToLabel, Label> {
    fn get_nodeindex(&self, rank : u32) ->  Option<&NodeIndex>;
    //
    fn get_label(&self, oldlabel : ToLabel) -> Option<&Label>;
} // end of trait IdMapper

impl<ToLabel, Label>  IdMap<ToLabel, Label> 
    where   Label   : LabelT ,
            ToLabel : Eq + Hash {

    pub fn new(ranktoidx : IndexMap::<u32, NodeIndex>, relabel : IndexMap<ToLabel, Label>) -> Self {
        IdMap{ranktoidx, relabel}
    }

}  // end of impl IdMap



impl <ToLabel, Label> IdMapper<ToLabel, Label>  for IdMap<ToLabel, Label>
        where     Label : LabelT ,
                ToLabel : Eq + Hash {

    /// get NodeIndex from rank in datafile
    fn get_nodeindex(&self, rank : u32) -> Option<&NodeIndex> {
        self.ranktoidx.get(&rank)
    } // end of get_nodeindex
    
    /// get new label from initial label in data file
    fn get_label(&self, oldlabel : ToLabel) -> Option<&Label> {
        self.relabel.get(&oldlabel)
    }  // end of get_label

} // end of impl block
