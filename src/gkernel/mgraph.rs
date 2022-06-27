//! mini implementation of a multigraph.
//! 
//! This multigraph is devoted to implement a nodesketch type algorithm
//! with hashing based on label transitions during edge traversal around a node
//! Nodes can have multiple disctrete lables to modelize multi communities membership
//! Edges can be directed or not and can have one discrete label.



/// A multigraph node
/// It must be possible to hash a label so it must satisfy Hash (possibly the Sig from probminhash if needed)
/// Each transition from a node to an edge will generate the 2-uples (l)
pub struct Mnode<NodeId, Label> {
    /// node unique identification
    id : NodeId, 
    ///
    labels : Vec<Label>,
    ///
    edges : Vec<Medge>
}  // end of Mnode


impl <NodeId, Label> Mnode<NodeId, Label> {

} // end of Mnode






/// An edge of our multigraph
/// An edge is characterized by a 2-uple of NodeId
pub struct Medge {
    /// The 2 nodes extremities of node
    /// By default the orientation is from nodes.0 to nodes.1
    nodes : (NodeId, NodeId),
    /// the label of an edge (expressing a kind of relation between 2 nodes)
    label : Label,
    /// directed or not
    directed : bool,
    /// edge weight, by default it is 1.
    weight : f32,
}  // end of Medge



pub struct Mgraph {
    /// nodes
    nodes: HashMap<NodeId, Mnode>,
    /// edges
    edges : HashMap<(Mnode, Mnode), Medge>,

} // end of Mgraph