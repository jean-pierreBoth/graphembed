//! This module gathers the embedding based on the paper:  
//! *NodeSketch : Highly-Efficient Graph Embeddings via Recursive Sketching KDD 2019*.  <https://dl.acm.org/doi/10.1145/3292500.3330951>  
//!    D. Yang,P. Rosso,Bin-Li, P. Cudre-Mauroux. 
//! 
//! It provides embedding for simple graphs without data attached to nodes or labels. To do embeddings with discrete data
//! attached to graph entities see the module gkernel [crate::embed::gkernel].  
//! The hashing strategy is based on Ertl's probminhash. (see the probminhash crate).  
//! It provides an extension for directed graph

pub mod params;

pub mod nodesketchsym;
pub mod nodesketchasym;

pub mod sla;
