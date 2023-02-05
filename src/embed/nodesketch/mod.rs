//! This module gathers the embedding based on the paper:  
//! *NodeSketch : Highly-Efficient Graph Embeddings via Recursive Sketching KDD 2019*.  <https://dl.acm.org/doi/10.1145/3292500.3330951>  
//!    D. Yang,P. Rosso,Bin-Li, P. Cudre-Mauroux. 
//! 
//! It is based on multi hop neighbourhood identification via sensitive hashing.
//! Instead of using **ICWS** for hashing we use the more recent algorithm **probminhash**. 
//! See  the paper [probminhash](https://arxiv.org/abs/1911.00675) and crate [probminhash](https://crates.io/crates/probminhash). 
//! The algorithm associates a probability distribution on neighbours of each point depending on edge weights and distance to the point.  
//! 
//! Then this distribution is hashed to build a (discrete) embedding vector consisting in nodes identifiers.  
//! The distance between embedded vectors is the Jaccard distance so we get
//! a real distance on the embedding space for the symetric embedding.  
//!
//! An extension of the paper is also implemented to get asymetric embedding for directed graph. 
//! The similarity is also based on the hash of sets (nodes going to or from) a given node but then the dissimilarity is no more a distance (no symetry and some discrepancy with the triangular inequality).
//!
//! It provides embedding for simple graphs without data attached to nodes or labels.  
//! To do embeddings with discrete data attached to graph entities see the module gkernel [crate::embed::gkernel].  
//! 


pub mod params;

pub mod nodesketchsym;
pub mod nodesketchasym;

pub use params::{NodeSketchParams};
pub use {nodesketchsym::NodeSketch, nodesketchasym::NodeSketchAsym};


pub mod sla;
