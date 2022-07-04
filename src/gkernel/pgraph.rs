//! interface towards petgraph::graph.
//! petgraph::graph permits multiple weighted edges between 2 nodes and Directed or Undirected graph

use std::hash::Hash;
use std::cmp::Eq;

use std::fmt::Display;


/// Our labels must satisfy.
pub trait LabelT : Eq + Hash + Clone + Default + Display {}


/// defines associated data to a Node.
/// A node can have many lables or be member of many communities
#[derive(Clone, Debug)]
pub struct Nweight<Nlabel> {
    /// memberships 
    labels : Vec<Nlabel>,
}


impl <Nlabel>  Nweight<Nlabel> 
    where Nlabel : LabelT {
    /// has_label ?
    pub fn has_label(&self, label : &Nlabel) -> bool {
        self.labels.iter().any(|l| l== label)
    }

    pub fn get_labels(&self) -> &[Nlabel] {
        &self.labels
    }
} // end of Nweight


//=================================================================================== 

/// Our edge label. Called Eweight as petgraph items attached to an entity is called a weight
pub struct Eweight<Elabel> {
    /// edge type/data
    label : Elabel,
    ///
    weight : f32
}

impl <Elabel> Eweight<Elabel> 
    where Elabel : LabelT {

    pub fn new(label : Elabel, weight : f32) -> Self {
        Eweight{ label, weight}
    }


    pub fn get_label(&self) -> &Elabel {
        &self.label
    }

    pub fn get_weight(&self) -> f32 {
        self.weight
    }

}  // end of Eweight