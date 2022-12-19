//!
//! module dedicated to density-friendly decomposition according to Danisch Chan Sozio _Large Scale decomposition via convex programming  2017
//!
//! It computes a decomposition of graph in blocks of vertices of decreasing density. 

/// Implements PAVA algorithm for isotonic regression.
pub  mod pava;

pub mod algodens;

/// Describe stable block results
pub mod stable;