//! This module gathers the embedding based on the paper:  
//! 
//! *NodeSketch : Highly-Efficient Graph Embeddings via Recursive Sketching KDD 2019*.  <https://dl.acm.org/doi/10.1145/3292500.3330951>  
//!    D. Yang,P. Rosso,Bin-Li, P. Cudre-Mauroux. 
//! 
//! It is based on probminhash from Ertl (see the probminhash crate).  
//! It provides an asymetric extension

pub mod params;

pub mod nodesketch;
pub mod nodesketchasym;

pub mod sla;
pub mod degrees;

pub const IN : u8 = 1;
pub const OUT : u8 = 0;