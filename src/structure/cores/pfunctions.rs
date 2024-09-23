//! various pfunctions implementation

#![allow(unused)]

use anyhow::anyhow;

use cpu_time::ProcessTime;
use std::time::SystemTime;

use num_traits::{float::*, FromPrimitive};

// synchronization primitive
use atomic::{Atomic, Ordering};
use parking_lot::RwLock;
use rayon::prelude::*;
use std::sync::Arc;

use indexmap::IndexSet;
use petgraph::graph::{DefaultIx, EdgeReference, Graph, IndexType, NodeIndex};
use petgraph::{visit::*, EdgeType, Undirected};

#[cfg_attr(doc, katexit::katexit)]
///
/// This function computes partial degree taking into account only neighbours restricted to a subset of vertices  
///
/// $$
/// p_{1}(node,vset) = deg(node,vset)
/// $$
/// with $ vset \subset V$ is the subset of vertices of V to which we restrict neighbours of $node$
pub fn p1<N, F, Ty, Ix>(
    graph: &Graph<N, F, Ty, Ix>,
    vset: &IndexSet<NodeIndex<Ix>>,
    node: NodeIndex<Ix>,
) -> f64
where
    Ty: EdgeType,
    Ix: IndexType,
{
    //
    let mut degree: usize = 0;
    let mut neighbors = graph.neighbors(node);
    for n in neighbors {
        // check if n is in vset
        if vset.get(&n).is_some() {
            degree += 1;
        }
    }
    // search neighbors of that are in vset
    degree as f64
} // end of p1

#[cfg_attr(doc, katexit::katexit)]
///
/// This function computes partial **incoming** degree taking into account only neighbours restricted to a subset of vertices.  
///   
/// If the graph is undirected, it is the same all edges from or to node argument
/// $$
/// p_{1}(node,vset) = indeg(node,vset)
/// $$
/// with $ vset \subset V$ is the subset of vertices of V to which we restrict neighbours of $node$
pub fn p2<N, F, Ty, Ix>(
    graph: &Graph<N, F, Ty, Ix>,
    vset: &IndexSet<NodeIndex<Ix>>,
    node: NodeIndex<Ix>,
) -> f64
where
    Ty: EdgeType,
    Ix: IndexType,
{
    //
    let mut degree: f64 = 0.;
    let mut edges = graph.neighbors_directed(node, petgraph::Incoming).detach();
    while let Some((edge, n)) = edges.next(graph) {
        if vset.get(&n).is_some() {
            degree += 1.;
        }
    }
    degree
} // end of p2

#[cfg_attr(doc, katexit::katexit)]
///
/// This function computes partial **outgoing** degree taking into account only neighbours restricted to a subset of vertices.  
///   
/// If the graph is undirected, it is the same all edges from or to node argument
/// $$
/// p_{1}(node,vset) = outdeg(node,vset)
/// $$
/// with $ vset \subset V$ is the subset of vertices of V to which we restrict neighbours of $node$
pub fn p3<N, F, Ty, Ix>(
    graph: &Graph<N, F, Ty, Ix>,
    vset: &IndexSet<NodeIndex<Ix>>,
    node: NodeIndex<Ix>,
) -> f64
where
    Ty: EdgeType,
    Ix: IndexType,
{
    //
    let mut degree: f64 = 0.;
    let mut edges = graph.neighbors_directed(node, petgraph::Outgoing).detach();
    while let Some((edge, n)) = edges.next(graph) {
        if vset.get(&n).is_some() {
            degree += 1.;
        }
    }
    degree
} // end of p3

#[cfg_attr(doc, katexit::katexit)]
///
/// This function computes sum of edge weight to neighbours in a subset of vertices.  
///
/// $$
/// p_{11}(node,vset) = \sum_{u \in N(node,vset)} w(node,u)
/// $$
/// with $vset  \subset V$ is the subset of vertices of V to which we restrict neighbours of $node$ and  
/// $w(v,u)$ is the weight of the edge from v to u.
pub fn p11<N, F, Ty, Ix>(
    graph: &Graph<N, F, Ty, Ix>,
    vset: &IndexSet<NodeIndex<Ix>>,
    node: NodeIndex<Ix>,
) -> f64
where
    Ty: EdgeType,
    Ix: IndexType,
    F: Float + FromPrimitive + std::ops::AddAssign,
{
    //
    let mut weight = F::zero();
    let mut neighbors = graph.neighbors(node).detach();
    while let Some((e, n)) = neighbors.next(graph) {
        // check if n is in vset
        if vset.get(&n).is_some() {
            // get edge weight
            weight += graph[e];
        }
    }
    //
    weight.to_f64().unwrap()
} // end of p11
