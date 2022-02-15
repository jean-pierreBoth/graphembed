//! describes embedder trait

use crate::embedding::*;

/// A trait to be able to manipulate embedder in a somewhat unified way if necessary
/// Comes at the cost of the Boxing of the result.
/// F is type of vector in which we embed, mostly f64, f32, usize
/// Useful just to make cross validation generic.

pub trait EmbedderT<F> {
    /// specific arguments of the embedder process
    type Params;
    ///
    fn embed(& mut self) -> Result<Box<dyn EmbeddingT<F>>, anyhow::Error>;
} // end of trait EmbedderT<F>