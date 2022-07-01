//! mini implementation of a multigraph.
//! 
//! This multigraph is devoted to an implementation of a nodesketch type algorithm
//! with hashing based on couple (origin node label, edge label) during edge traversal toward a node.   
//! Nodes can have multiple discrete labels to modelize multi communities membership and various relations
//! between nodes.     
//! Edges can be directed or not and can have at most one discrete label, but there can be many edges between 2 given nodes
//! as log as the labels of edge are unique.   
//! Edge can also have a weight, by default set 1.
//! 
//! We should get both an embedding of each node in terms of (Nlabel, Elabel) and a global graph summary vector  
//! 


use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Iter;

use std::hash::Hash;
use std::cmp::Eq;

use std::fmt::Display;

/// all labels must satisfy this trait
pub trait LabelT : Eq + Hash + Clone + Display {}

/// all identificators lust satisfy this trait
pub trait IdT : Eq + Hash + Copy + Display {}


// a node must have an id and has possibly multiple labels
pub trait NodeT<NodeId, Nlabel> {
    fn get_id(&self) -> NodeId;
    ///
    fn get_labels(&self) ->  &HashSet<Nlabel>;
} // end of trait NodeT


/// A multigraph node
/// It must be possible to hash a label so it must satisfy Hash (possibly the Sig from probminhash if needed).
/// If the node has no label (yet) the lables hashset can be empty (useful for community detection)  
/// Each transition from a node to an edge will generate the 2-uples (label, edge_label) for each label of the node.
/// We want Mnode to be Sync so NodeId, Nlabel and Medge should be Sync
/// Nlabel and Elabel must satisfy to provide default initialization of sketched values
pub struct Mnode<'a, NodeId, Nlabel, Elabel> 
    where NodeId : Copy,
          Elabel : Clone {
    /// node unique identification
    id : NodeId, 
    ///
    labels : HashSet<Nlabel>,
    /// references to edges as stored in the Mgraph structure with a lifetime 'a (larger than )
    edges : Vec<&'a Medge<NodeId, Elabel> >
}  // end of Mnode



/* impl <'a , NodeId, Nlabel, Medge> NodeT<NodeId, Nlabel> for Mnode<'a, NodeId, Nlabel, Medge> 
    where NodeId : Copy ,
          Medge  : Clone {

    ///
    fn get_id(&self) -> NodeId {
        self.id.clone()
    }
    /// get all the labels of this node
    fn get_labels(&self) -> &HashSet<Nlabel> {
        &self.labels
    }

}  // 
 */


impl <'a, NodeId, Nlabel, Elabel> Mnode<'a, NodeId, Nlabel, Elabel> where 
    Nlabel : LabelT,
    NodeId : Eq + Copy + Hash,
    Elabel : LabelT,
    Medge<NodeId, Elabel> : Clone  {

    ///
    fn get_id(&self) -> NodeId {
        self.id.clone()
    }
    /// get all the labels of this node
    fn get_labels(&self) -> &HashSet<Nlabel> {
        &self.labels
    }
    ///
    pub fn get_edges(&self) -> &Vec<&'a Medge<NodeId, Elabel> > {
        &self.edges
    }
    /// 
    pub fn new(id : NodeId, labels: HashSet<Nlabel>, edges: Vec<&'a Medge<NodeId, Elabel> >) -> Self {
        Mnode{id , labels, edges}
    }
    /// add an edge
    pub fn add_edge(&mut self, edge: &'a Medge<NodeId, Elabel>) {
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
    fn get_label(&self) -> &Elabel;
    ///
    fn is_directed(&self) -> bool;
    ///
    fn get_nodes(&self) -> &(NodeId, NodeId);
    ///
    fn get_weight(&self) -> f32;
}  // end of impl EdgeT


/// An edge of our multigraph.
/// An edge is characterized by a 2-uple of NodeId.
/// If the edge has no specific label attached, its rank can be used.
/// We want Medge to be Sync so NodeId, Elabel should be Sync
#[derive(Clone)]
pub struct Medge<NodeId, Elabel>
    where NodeId : Copy, 
          Elabel : Clone {
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


/// an alias to simplify notations


impl <NodeId, Elabel> EdgeT<NodeId, Elabel> for  Medge<NodeId, Elabel> 
        where NodeId : IdT,
              Elabel : LabelT {
    ///
    fn get_nodes(&self) -> &(NodeId, NodeId) {
        &self.nodes
    }
    /// get th (unique) lable of this edge
    fn get_label(&self) -> &Elabel {
        &self.label
    }
    ///
    fn is_directed(&self) -> bool {
        self.directed
    }
    ///
    fn get_weight(&self) -> f32 {
        self.weight
    }
}  // end of EdgeT implementation




impl <NodeId, Elabel> Medge<NodeId, Elabel>
    where NodeId : Eq + Hash + Copy,
          Elabel : Eq + Hash + Clone  {
    ///
    pub fn new( nodes: (NodeId, NodeId), label : Elabel, directed : bool, weight : f32) -> Self  {
        Medge{nodes, label, directed, weight}
    }

    pub fn get_nodes(&self) -> &(NodeId, NodeId) {
        &self.nodes
    }
    /// get th (unique) lable of this edge
    fn get_label(&self) -> &Elabel {
        &self.label
    }
    ///
    pub fn is_directed(&self) -> bool {
        self.directed
    }
    ///
    pub fn get_weight(&self) -> f32 {
        self.weight
    }
} // end of impl <NodeId, ELabel> Medge



//==============================================================================

pub type EdgeSet<Elabel, NodeId> = HashMap<Elabel, Medge<NodeId, Elabel> >;


pub struct Mgraph<'a, NodeId, Nlabel, Elabel> 
    where NodeId : Copy, 
          Elabel : Clone,
          Nlabel : Clone {
    /// nodes
    nodes: HashMap<NodeId, Mnode<'a, NodeId, Nlabel, Elabel> >,
    /// edges. As we can have many edges for a couple of Node
    edges : HashMap<(NodeId, NodeId), EdgeSet<Elabel, NodeId> >,
    //
//    _phantom_n :  std::marker::PhantomData<Nlabel>,
    //
//    _phantom_e :  std::marker::PhantomData<Elabel>,
} // end of Mgraph


impl <'a, NodeId, Nlabel, Elabel> Default for Mgraph<'a, NodeId, Nlabel, Elabel> 
        where   NodeId : IdT,
                Elabel : LabelT, 
                Nlabel : LabelT
            {
    //
    fn default() -> Self {
        Mgraph{nodes : HashMap::<NodeId, Mnode<NodeId, Nlabel, Elabel> >::new(), edges: HashMap::<(NodeId, NodeId), EdgeSet<Elabel, NodeId> >::new()}
    }
} // end of impl Default




impl <'a, NodeId, Nlabel, Elabel> Mgraph<'a, NodeId, Nlabel, Elabel> 
    where   NodeId : IdT,
            Elabel : LabelT, 
            Nlabel : LabelT  {

    /// get number of nodes
    pub fn get_nb_nodes(&self) -> usize {
        self.nodes.len()
    } // end of get_nb_nodes


    /// get an iterator over nodes
    pub fn get_node_iter(&self) -> Iter<NodeId, Mnode<NodeId, Nlabel, Elabel>> {
        self.nodes.iter()
    } // end of get_node_iter


    /// get an iterator over edge set by couple of nodes
    /// Recall that if an edge is oriented it goes from first node to second node
    pub fn get_edgeset_iter(&self) -> Iter<(NodeId, NodeId), EdgeSet<Elabel, NodeId>> {
        self.edges.iter()
    }

    /// adding a node.
    pub fn add_node(&mut self, node : Mnode<'a, NodeId, Nlabel, Elabel>) {
        let nodeid = node.get_id();
        let already = self.nodes.insert(nodeid, node);
        if already.is_some() {
            log::error!("could not insert node id {}, already present", &nodeid);
            std::process::exit(1);
        }
    } // end of add_node



    /// return an option on an edge given nodes and edge label, None if there is no such edge
    pub fn get_edge_with_label(&self, nodes: &(NodeId, NodeId), label : &Elabel) -> Option<&Medge<NodeId, Elabel> > {
        let hmap  = self.edges.get(nodes);
        match hmap {
            Some(hmap) => {
                return hmap.get(label);
            }
            None => { return None;}
        }
    } // end of get_edge


    #[allow(unused)]
    pub(crate) fn get_edge_set(&self, nodes: &(NodeId, NodeId)) -> Option<&HashMap<Elabel, Medge<NodeId, Elabel> > >  {
        self.edges.get(nodes)
    }


    /// adding an edge.
    // TODO The dispatching to nodes is to be done WHEN?
    pub fn add_edge(&mut self, edge : Medge<NodeId, Elabel>) {
        let edge_nodes = edge.get_nodes();
        let edges = &mut self.edges;
        if let Some(hmap) = edges.get_mut(&edge_nodes) {
            let elabel = edge.get_label();
            if let Some(_) = hmap.get(edge.get_label()) {
                log::error!("could not insert already present edge ({}, {}) with label {}", &edge_nodes.0, &edge_nodes.1, &elabel);
                std::process::exit(1);                      
            }
            else { // we insert in hmap
                hmap.insert(elabel.clone(), edge.clone());
                return;
//                let node_0 = self.nodes.get_mut(&edge_nodes.0).unwrap();
 //               node_0.add_edge(edge_ref);
            }            
        }
        else {
            let elabel = edge.get_label();
            let mut ehashmap = HashMap::<Elabel, Medge<NodeId, Elabel> >::new();
            ehashmap.insert(elabel.clone(), edge.clone());
            edges.insert(*(edge_nodes), ehashmap);
            return;
        }

        // retrieve a reference to the inserted edge
        // Dispatch to concerned nodes. So Nodes must have been declared before edges referencing it.
//        let nodes = &mut self.nodes;
//        let node_0 = nodes.get_mut(&edge_nodes.0).unwrap();
//        node_0.add_edge(edge_ref);


    }  // end of add_edge

    // loop on edges, and update nodes with related edges
    // the &'a implies that edge lifetime transferred into nodes is same as graph lifetime
    fn dispatch_edges(&'a mut self) {
        for (nodes, edgeset) in self.edges.iter() {
            for (_label, edge) in edgeset {
                self.nodes.get_mut(&nodes.0).unwrap().add_edge(edge);
                self.nodes.get_mut(&nodes.1).unwrap().add_edge(edge);

                
            }

        }


    } // end of dispatch_edges

} // end of Mgraph






//==========================================================================
