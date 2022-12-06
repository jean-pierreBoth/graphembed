//! various pfunctions implementation

#![allow(unused)]

use anyhow::{anyhow};



use std::time::{SystemTime};
use cpu_time::ProcessTime;

use num_traits::{float::*, FromPrimitive};

// synchronization primitive
use std::sync::{Arc};
use parking_lot::{RwLock};
use atomic::{Atomic, Ordering};
use rayon::prelude::*;

use petgraph::graph::{Graph, EdgeReference, NodeIndex, DefaultIx};
use petgraph::{Undirected, visit::*};

#[cfg_attr(doc, katexit::katexit)]
/// This function computes $p1(v) = deg(v,vset), where vset \subset V$
pub fn p1<'a, N, F, Ty, Ix>(graph : &'a Graph<N, F, Ty, Ix>, vset : &[NodeIndex<Ix>], node : NodeIndex<Ix>) -> f64 {
    // search neighbors of that are in vset
    std::panic!("not yet implemented");
} // end of p1