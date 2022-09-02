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


use probminhash::probminhasher::*;
/// Our labels must satisfy:
/// For having String as possible labels we need Clone.
/// To hash strings with sha2 crate we must get something equivalent to AsRef<[u8]>
/// This is provided by Sig (and is required by Probminhash3sha which do not need copy on items hashed)
pub trait LabelT : Send + Sync + Eq + Hash + Clone + Default + std::fmt::Debug + sig::Sig {} 

impl LabelT for u8 {}
impl LabelT for u16 {}
impl LabelT for u32 {}
impl LabelT for u64 {}
impl LabelT for i32 {}
impl LabelT for i16 {}
impl LabelT for String {}


/// A label type encoding a couple of Node label and edge label representing a transiiton from/to a node via a labelled edge

#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct NElabel<Nlabel,Elabel>(pub(crate) Nlabel, pub(crate) Elabel);


impl<Nlabel, Elabel>  sig::Sig for NElabel<Nlabel, Elabel> where 
    Nlabel : LabelT, Elabel : LabelT  {
    fn get_sig(&self) -> Vec<u8> {
        let mut s = self.0.get_sig().clone();
        s.append(&mut self.1.get_sig().clone());
        return s;
    }
} // end of impl Sig for NElabel



/// defines associated data to a Node.
/// A node can have many lables or be member of many communities
#[derive(Clone, Debug)]
pub struct Nweight<Nlabel> {
    /// memberships 
    labels : Vec<Nlabel>,
}


impl <Nlabel>  Nweight<Nlabel> 
    where Nlabel : LabelT {
    ///
    pub fn new(labels : Vec<Nlabel>) -> Self {
        Nweight{labels : labels}
    }
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
/// The edge has a label attached and a f32 weight.
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

    // retrieve the label of the edge
    pub fn get_label(&self) -> &Elabel {
        &self.label
    }

    /// retrieves edge weight
    pub fn get_weight(&self) -> f32 {
        self.weight
    }

}  // end of Eweight

/// Data associated to an edge should satisfy Default and so Label should satisfy default.
impl <Elabel> Default for Eweight<Elabel> 
    where Elabel : LabelT {
    fn default() -> Self {
        Eweight{ label : Elabel::default(), weight: 1.}
    }
}

//=============================================================================


pub trait HasNweight<Nlabel:LabelT> {
    fn get_nweight(&self) -> &Nweight<Nlabel>;
}

pub trait HasEweight<Elabel:LabelT> {
    fn get_eweight(&self) -> &Eweight<Elabel>;
}