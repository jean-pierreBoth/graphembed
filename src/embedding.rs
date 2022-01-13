//! Describes the embedding vectors. 
//! 
//! Basic symetric Embedding are described by an Array2<F> where F is f32 or f64.  
//! The type F is described by the constraints  F : Float + Lapack + Scalar  + ndarray::ScalarOperand + sprs::MulAcc.  
//! Each row corresponds to a node.
//! 
//! For an asymetric or directed graph, an asymetric embedding distinguishes for each node, its target role from its source role.
//! So we need 2 matrices to represent the embedding.
//! 
//! 


use num_traits::float::*;

use ndarray_linalg::{Scalar};
use ndarray::{Array2};


pub struct Embedding<F:Scalar+Float>(Array2<F>);

impl<F:Scalar+Float> Embedding<F> {

    pub(crate) fn new( arr : Array2<F>) -> Self {
        Embedding{0 : arr}
    }

    /// get dimension of embedding. (row size of Array)
    pub fn get_dimension(&self) -> usize {
        self.0.dim().1
    }
}  // end of impl Embedding



//============================================


/// Asymetric embedding representation
pub struct EmbeddingAsym<F:Scalar+Float> {
    /// source node representation
    source : Array2<F>,
    /// target node representation
    target : Array2<F>
} // end of struct EmbeddingAsym


impl <F:Scalar+Float> EmbeddingAsym<F> {

    pub(crate) fn new(source : Array2<F>, target : Array2<F>) -> Self {
        assert_eq!(source.dim().0, target.dim().0);
        assert_eq!(source.dim().1, target.dim().1);
        EmbeddingAsym{source, target}
    }

    pub fn get_source(&self) -> &Array2<F> {
        &self.source
    }

    pub fn get_target(&self) -> &Array2<F> {
        &self.target
    }
 
    /// get dimension of embedding. (row size of Array)
    pub fn get_dimension(&self) -> usize {
        self.source.dim().1
    }
} // end of impl block for EmbeddingAsym