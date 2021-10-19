//! this file contains implementation of icws as
//! described in
//! *Improved Consistent Weighted Sampling Revisited*
//! Wu,Li, Chen, Zhang, Yu 2017. (Wu-arxiv-2017)<https://arxiv.org/abs/1706.01172>
//!



use rand::distributions::{Distribution,Uniform};
use rand_distr::Exp1;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;