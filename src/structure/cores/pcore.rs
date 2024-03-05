//! implements p-core determination for p-functions (see associated file)
//! We implement algo 4 of Batagelj-Zaversnik paper

#![allow(unused)]

use std::collections::BinaryHeap;

use anyhow::anyhow;

use indexmap::IndexSet;

use ordered_float::OrderedFloat;
use petgraph::graph::{Graph, NodeIndex};

pub fn pcore4<N, E, Ty, Ix>(
    graph: &Graph<N, E, Ty, Ix>,
    pfunction: fn(&Graph<N, E, Ty, Ix>, &IndexSet<NodeIndex<Ix>>, NodeIndex<Ix>) -> f64,
) {
    let min_heap = BinaryHeap::<OrderedFloat<f64>>::new();
} // end of pcore4
