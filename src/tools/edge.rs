//! just an edge


/// for an edge for i to j
/// we say that that the edge is IN for j and out for i.
pub const OUT : u8 = 0;
pub const IN : u8 = 1;
pub const INOUT : u8 = 2;

/// codes for orientation of edge. INOUT is for symetric edge.
pub enum EdgeDir {
    OUT,
    IN,
    INOUT
}


// local edge type corresponding to node1, ndde2 , distance from node1 to node2
#[derive(Copy, Clone, Debug)]
pub struct Edge(pub usize, pub usize, pub f64);