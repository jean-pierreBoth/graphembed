//! This module is devoted to graph decomposition (in order to do graph embedding validation)
//!
//! Algorithms implemented (or reimplemented) are:
//! 
//! - for undirected graph:
//! 
//!   - Danisch Chan Sozio _Large Scale decomposition via convex programming_  [2017](https://dl.acm.org/doi/10.1145/3038912.3052619)
//!
//!   - Batagelj Zaversnik _Fast algorithms for determining generalized core in networks_ [2011](https://link.springer.com/article/10.1007/s11634-010-0079-y)

//! 
//!  See also:  
//!     - _Tatti Gionis Density-friendly graph decomposition_ [2015](https://dl.acm.org/doi/10.1145/2736277.2741119)  
//!     - _Tatti Density-friendly graph decomposition_ [2019](https://arxiv.org/abs/1904.03467)
//! 

// - for directed graph:
//     - Giatsidis Thilikos Vazirgiannis D_cores: Measuring Collaboration in Directed graphs based on degeneracy 2013.
//

// implements generalized core decomposition according to Batagelj Zaversnik paper
// pub mod kcore;

/// density decomposition according to Danisch Chan Sozio _Large Scale decomposition via convex programming 2017_
pub mod density;

/// core decompositions according to Batagelj Zaversnik 2011
pub mod cores;