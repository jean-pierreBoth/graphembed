//! This module implements some validation tools
//!  We use:
//! - the standard link prediction [link::estimate_auc()] based and a centric auc [link::estimate_centric_auc]
//! - A decomposition in stable blocks to evaluate preservation of "communities" [anndensity]

pub mod link;
/// references
/// - The Link-Prediction Problem for Social Networks
///         Liben-Nowell Kleinberg  2007 Journal AMS for Info. Science and Technology
///
/// - Link Prediction in complex Networks : A survey
///         Lü, Zhou. Physica 2011
///
/// - A Survey of Link Prediction in Complex Networks
///          Martinez, Berzal, Cubero 2016
///
/// - Path‑based extensions of local link prediction methods for complex networks
///     Aziz, Gul Nature Scientific reports 2020.
pub mod linkparams;

pub mod anndensity;
