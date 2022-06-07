//! This module partially implements Hope asymetric embedder
//! 
//! Only the Adamic Adar similarity between nodes is operational at present time.  
//! It builds upon approximated svd implemented in the crate annembed.

pub mod gsvd;
pub mod randgsvd;

pub mod hope;

pub mod orderingf;