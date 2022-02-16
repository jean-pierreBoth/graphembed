//! describes embedder trait to be able to manipulate embedder in a somewhat unified way

use crate::embedding::EmbeddingT;

/// The trait Embedder is something that has as output
/// something satisfying the trait EmbeddingT<F>.  
/// F is the type contained in embedded vectors , mostly f64, f32, usize
/// Useful just to make cross validation generic.

pub trait EmbedderT<F>  where Self::Output : EmbeddingT<F> {
    type Output;
    ///
    fn embed(& mut self) -> Result< Self::Output, anyhow::Error>;
} // end of trait EmbedderT<F>