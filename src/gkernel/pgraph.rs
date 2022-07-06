//! This module describes Edge and Node data we use in petgraph in an implementation of a nodesketch type algorithm.
//! Nodes can have multiple discrete labels to modelize multi communities membership and various relations
//! between nodes.     
//! Edges can be directed or not and can have at most one discrete label, but there can be many edges between 2 given nodes
//! as log as the labels of edge are unique.   
//! Edge can also have a weight, by default set 1.
//! 
use std::hash::Hash;
use std::cmp::Eq;

// use std::fmt::Display;


use probminhash::probminhasher::sig;
/// Our labels must satisfy:
/// For having String as possible labels we need Clone.
/// To hash strings with sha2 crate we must get something equivalent to AsRef<[u8]>
/// This is provided by Sig (and is required by Probminhash3sha which do not need copy on items hashed)
pub trait LabelT : Send + Sync + Eq + Hash + Clone + Default + std::fmt::Debug + sig::Sig {} 


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
        self.labels.iter().any(|l| l == label)
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