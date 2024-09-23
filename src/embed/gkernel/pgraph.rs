//! This module describes Edge and Node data we use in petgraph in an implementation of a nodesketch type algorithm.
//!
//! Nodes can have multiple discrete labels to modelize multi communities membership and various relations
//! between nodes.    
//! Edges can be directed or not and can have at most one discrete label, but there can be many edges between 2 given nodes
//! Edge can also have a weight, by default set 1.
//!
use std::cmp::Eq;
use std::hash::Hash;

// use std::fmt::Display;

use probminhash::probminhasher::*;
/// Our labels must satisfy this trait.
///
/// - For having String as possible labels we need Clone.
///
/// - To hash strings or Vectors with sha2 crate we must be able to associate to labels something statisfying a Vec\<u8\>.  
///     This is provided by Sig (and is required by Probminhash3sha which do not need copy on items hashed)
pub trait LabelT: Send + Sync + Eq + Hash + Clone + Default + std::fmt::Debug + sig::Sig {}

impl LabelT for u8 {}
impl LabelT for u16 {}
impl LabelT for u32 {}
impl LabelT for u64 {}
impl LabelT for i32 {}
impl LabelT for i16 {}
impl LabelT for String {}

/// A label type encoding a couple of Node label and edge label representing a transiiton from/to a node via a labelled edge

#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct NElabel<Nlabel, Elabel>(pub(crate) Nlabel, pub(crate) Elabel);

impl<Nlabel, Elabel> sig::Sig for NElabel<Nlabel, Elabel>
where
    Nlabel: LabelT,
    Elabel: LabelT,
{
    fn get_sig(&self) -> Vec<u8> {
        let mut s = self.0.get_sig().clone();
        s.append(&mut self.1.get_sig().clone());
        s
    }
} // end of impl Sig for NElabel

/// defines associated data to a Node.
/// A node can have many (discrete) labels (may be participate in many communities).
#[derive(Clone, Debug)]
pub struct Nweight<Nlabel> {
    /// memberships
    labels: Vec<Nlabel>,
}

impl<Nlabel> Nweight<Nlabel>
where
    Nlabel: LabelT,
{
    //
    pub fn new(labels: Vec<Nlabel>) -> Self {
        Nweight { labels }
    }
    /// has_label ?
    pub fn has_label(&self, label: &Nlabel) -> bool {
        self.labels.iter().any(|l| l == label)
    }

    pub fn get_labels(&self) -> &[Nlabel] {
        &self.labels
    }
} // end of Nweight

//===================================================================================

/// Our edge label, called Eweight as petgraph items attached to an entity is called a weight.  
/// The edge has a f32 weight which defaults to 1.
/// Edges may have discrete labels attached to it, initialized via the label option argument.
pub struct Eweight<Elabel> {
    /// edge type/data
    label: Option<Elabel>,
    //
    weight: f32,
}

impl<Elabel> Eweight<Elabel>
where
    Elabel: LabelT,
{
    pub fn new(label: Option<Elabel>, weight: f32) -> Self {
        Eweight { label, weight }
    }

    // retrieve the label of the edge
    pub fn get_label(&self) -> Option<&Elabel> {
        self.label.as_ref()
    }

    /// retrieves edge weight
    pub fn get_weight(&self) -> f32 {
        self.weight
    }
} // end of Eweight

/// Data associated to an edge should satisfy Default and so  Eweight\<Elabel\> should satisfy Default.
impl<Elabel> Default for Eweight<Elabel>
where
    Elabel: LabelT,
{
    fn default() -> Self {
        Eweight {
            label: None,
            weight: 1.,
        }
    }
}

//=============================================================================

/// A structure defining a node must implement this trait. See examples in gkernel::exio
pub trait HasNweight<Nlabel: LabelT> {
    fn get_nweight(&self) -> &Nweight<Nlabel>;
}

/// A structure defining an edge must implement this trait. See examples in gkernel::exio
pub trait HasEweight<Elabel: LabelT> {
    fn get_eweight(&self) -> &Eweight<Elabel>;
}
