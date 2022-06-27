//! mini implementation of a multigraph.
//! 
//! This multigraph is devoted to an implementation of a nodesketch type algorithm
//! with hashing based on label transitions during edge traversal around a node.  
//! Nodes can have multiple discrete labels to modelize multi communities membership.   
//! Edges can be directed or not and can have at most one discrete label. Edge can also have a weight, by default set 1.
//! 
//! We should get both an embedding of each node and a global graph summary vector  
//! 


use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::cmp::Eq;

use std::fmt::Display;

use std::marker::PhantomData;
// a node must have an id and has possibly multiple labels
pub trait NodeT<NodeId, Nlabel> {
    fn get_id(&self) -> NodeId;
    ///
    fn get_labels(&self) ->  &Vec<Nlabel>;
}
/// A multigraph node
/// It must be possible to hash a label so it must satisfy Hash (possibly the Sig from probminhash if needed).  
/// Each transition from a node to an edge will generate the 2-uples (label, edge_label) for each label of the node.
pub struct Mnode<NodeId, Nlabel, Medge> {
    /// node unique identification
    id : NodeId, 
    ///
    labels : HashSet<Nlabel>,
    ///
    edges : Vec<Medge>
}  // end of Mnode




impl <NodeId, Nlabel, Medge> Mnode<NodeId, Nlabel, Medge> where 
    Nlabel : Hash + Eq + Clone,
    NodeId : Eq + Copy {
    ///
    pub fn get_id(&self) -> NodeId {
        self.id
    }
    /// get all the labels of this node
    pub fn get_labels(&self) -> &HashSet<Nlabel> {
        &self.labels
    }
    ///
    pub fn get_edges(&self) -> &Vec<Medge> {
        &self.edges
    }
    /// 
    pub fn new(id : NodeId, labels: HashSet<Nlabel>, edges: Vec<Medge>) -> Self {
        Mnode{id , labels, edges}
    }
    /// add an edge
    pub fn add_edge(&mut self, edge: Medge) {
        self.edges.push(edge)
    }
    ///
    pub fn has_label(&self, label : &Nlabel) -> bool {
        self.labels.contains(label)
    }
    /// add label to the node, return true if label was not already present
    pub fn add_label(&mut self, label : &Nlabel) -> bool {
        self.labels.insert(label.clone())
    }
} // end of Mnode



//=============================================================================


pub trait EdgeT<NodeId, Elabel> {
    ///
    fn get_label(&self) -> Elabel;
    ///
    fn is_directed(&self) -> bool;
    ///
    fn get_nodes(&self) -> (NodeId, NodeId);
}
/// An edge of our multigraph.
/// An edge is characterized by a 2-uple of NodeId.
/// If the edge has no specific label attached, its rank can be used.
pub struct Medge<NodeId, Elabel>
    where NodeId : Eq, 
          Elabel : Eq + Hash {
    /// The 2 nodes extremities of node
    /// By default the orientation is from nodes.0 to nodes.1
    nodes : (NodeId, NodeId),
    /// the label of an edge (expressing a kind of relation between 2 nodes)
    label : Elabel,
    /// directed or not
    directed : bool,
    /// edge weight, by default it is 1.
    weight : f32,
}  // end of Medge




impl <NodeId, Elabel> Medge<NodeId, Elabel>
    where NodeId : Eq,
          Elabel : Eq + Hash  {
    ///
    pub fn new( nodes: (NodeId, NodeId), label : Elabel, directed : bool, weight : f32) -> Self  {
        Medge{nodes, label, directed, weight}
    }
    ///
    pub fn get_nodes(&self) -> &(NodeId, NodeId) {
        &self.nodes
    }
    /// get th (unique) lable of this edge
    pub fn get_label(&self) -> &Elabel {
        &self.label
    }
    ///
    pub fn get_weight(&self) -> f32 {
        self.weight
    }
    ///
    pub fn directed(&self) -> bool {
        self.directed
    }
} // end of impl <NodeId, ELabel> Medge



//==============================================================================


pub struct Mgraph<NodeId, Nlabel, Mnode, Medge, Elabel> {
    /// nodes
    nodes: HashMap<NodeId, Mnode>,
    /// edges. As we can have many edges for a couple of Node
    edges : HashMap<(NodeId, NodeId), HashMap<Elabel, Medge> >,
    ///
    _phantom_n :  std::marker::PhantomData<Nlabel>,
    ///
    _phantom_e :  std::marker::PhantomData<Elabel>,
} // end of Mgraph


impl <NodeId, Mnode, Nlabel, Medge, Elabel> Default for Mgraph<NodeId, Nlabel, Mnode, Medge, Elabel> {
    fn default() -> Self {
        Mgraph{nodes : HashMap::<NodeId, Mnode>::new(), edges: HashMap::<(NodeId, NodeId), HashMap<Elabel, Medge>>::new(), 
            _phantom_n : PhantomData, _phantom_e :  PhantomData}
    }
} // end of impl Deafault




impl <NodeId, Nlabel, Mnode, Medge, Elabel> Mgraph<NodeId, Nlabel, Mnode, Medge, Elabel> 
    where   Mnode : NodeT<NodeId, Nlabel>, 
            NodeId : Eq + Hash + Copy + Display,
            Elabel : Eq + Hash,
            Medge : EdgeT<NodeId, Elabel> {
    /// adding a node
    pub fn add_node(&mut self, node : Mnode) {
        let nodeid = node.get_id();
        let already = self.nodes.insert(nodeid, node);
        if already.is_some() {
            log::error!("could not insert node id {}", &nodeid);
            std::process::exit(1);
        }
    } // end of add_node



    /// return an option on an edge given nodes and edge label, None if there is no such edge
    pub fn get_edge(&self, nodes: &(NodeId, NodeId), label : &Elabel) -> Option<&Medge> {
        let hmap  = self.edges.get(nodes);
        match hmap {
            Some(hmap) => {
                return hmap.get(label);
            }
            None => { return None;}
        }
    } // end of get_edge


    /// adding an edge
    pub fn add_edge(&mut self, edge : Medge) {
        let nodes = edge.get_nodes();
        // must check if we have this edge
    }  // end of add_edge


} // end of Mgraph
