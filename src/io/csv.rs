//! Construct or dump a (small) graph from data from a csv

use csv;

use std::fmt::{Debug};

use std::path::{Path, PathBuf};
use std::fs::{OpenOptions, File};
use std::io::{Write,BufWriter, BufReader};

use petgraph::graph::*;

// From csv file we have nodes indexed by u32, edge multiplicity as f32,  edge can be negative 
// (see bitcoin files  https://snap.stanford.edu/data/index.html)
//
// https://snap.stanford.edu/data/index.html
// p2p-Gnutella09.txt  peer to peer directed unweighted
// ca-GrQc.txt  collaboration network undirected

/// Ty is Directed (default) or UnDirected
/// Ix is the node and edge index type , default u32
/// N and E in Grap<N,E,Ty,Ix are data associated to node and edge respectively>
/// instantiate with UnDirected for undirected graph
/// 
pub fn from_csv<Ty, Ix>(filename : &str) -> Option<Graph<Ty,Ix> > {
    None
} // end of from_csv