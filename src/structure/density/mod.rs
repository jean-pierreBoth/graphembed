//!   
//! Danisch Chan Sozio _Large Scale decomposition via convex programming_  [2017](https://dl.acm.org/doi/10.1145/3038912.3052619)
//!
//! It computes a decomposition of graph in blocks of vertices of decreasing density.   
//! The PAVA algorithm is a build block of stable decomposition but could be of interest by itself

/// Implements PAVA algorithm for isotonic regression.
pub  mod pava;
pub use pava::{IsotonicRegression, PointIterator};


pub mod algodens;
pub use stable::*;
pub use algodens::{approximate_decomposition};
/// Describe stable block results
pub mod stable;