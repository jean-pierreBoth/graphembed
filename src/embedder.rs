//! describes embedder trait


/// A trait to be able to manipulate embedder in a somewhat unified way if necessary
/// Comes at the cost of the Boxing of the result.
/// F is type of vector in which we embed, mostly f64, f32, usize
/// Useful just to make cross validation generic.

pub trait EmbedderT<F> {
    type Output;
    ///
    fn embed(& mut self) -> Result< Self::Output, anyhow::Error>;
} // end of trait EmbedderT<F>