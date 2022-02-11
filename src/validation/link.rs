//! a simple link prediction to test embeddings


/// The scoring function for similarity is based upon the distance relative to the embedding being tested
/// Jaccard for nodesketch and L2 for Atp
/// We first implement precision measure as described in 
///       - Link Prediction in complex Networks : A survey
///             Lü, Zhou. Physica 2011