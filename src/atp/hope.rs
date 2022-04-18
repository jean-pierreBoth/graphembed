//! Asymetric Graph Embedding
//! Based on the paper:
//!     Asymetric Transitivity Preserving Graph Embedding 
//!     Ou, Cui Pei, Zhang, Zhu   in KDD 2016
//!  See  [atp](https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf)
//! 
//! Implements only embedding built from Adamic Adar node representation.
//! 
//! The type F is supposed to be f32 or f64 and is constrained to satisfy whatever is expected for floats
//! 


use anyhow::{anyhow};

use ndarray::{Array1, Array2, ArrayView1};

use ndarray_linalg::{Scalar, Lapack};

use std::time::{SystemTime};
use cpu_time::ProcessTime;

use num_traits::float::*; 
// use num_traits::cast::FromPrimitive;

//use sprs::prod;
use sprs::{CsMat, TriMatI, TriMatBase};

use crate::tools::{degrees::*};

use annembed::tools::svdapprox::{MatRepr, MatMode, RangeApproxMode, SvdApprox, SvdResult};

use super::randgsvd::{GSvdApprox, GSvdResult};
use super::orderingf::*;
use crate::embedding::{EmbeddedAsym, EmbedderT};

/// The dissimilarity corresponding to hope. Note that it is not a distance, nor is it guaranteed to be positive.
/// Basically it is the opposite of the similarity estimated (and constructed in the Hope matrix)
pub fn hope_distance<F>(v1:&ArrayView1<F>, v2 : &ArrayView1<F>) -> f64 
    where F : Float + Scalar + Lapack {
    assert_eq!(v1.len(), v2.len());
    let dist2 = v1.iter().zip(v2.iter()).fold(F::zero(), |acc, v| acc + (*v.0 * *v.1));
    1.0 - dist2.to_f64().unwrap()
}


/// The distance corresponding to hope embedding. In fact it is Cosine
pub fn hope_distance_cos<F>(v1:&ArrayView1<F>, v2 : &ArrayView1<F>) -> f64 
    where F : Float + Scalar + Lapack {
    assert_eq!(v1.len(), v2.len());
    let dist = v1.iter().zip(v2.iter()).fold((F::zero(), F::zero(), F::zero()) , 
            |acc, v| (acc.0 + *v.0 * *v.0, acc.1 + *v.1 * *v.1, acc.2 + *v.0 * *v.1));
    //
    if dist.0 > F::zero() && dist.1 > F::zero() {
        let cos = dist.2/ (num_traits::Float::sqrt(dist.0 * dist.1));
        1.0 - cos.to_f64().unwrap()
    }
    else{
        1.
    }
} // end of jaccard



/// To specify if we run with Katz index or in Rooted Page Rank or Adamic Adar (a.k.a Resource Allocator)
#[derive(Copy,Clone, Debug)]
pub enum HopeMode {
    /// Katz index mode
    KATZ,
    /// Rooted Page Rank
    RPR,
    /// Adamic Adar or Resource allocator
    ADA,
} // end of HopeMode



#[derive(Copy, Clone, Debug)]
pub struct HopeParams {
    /// describe mode
    hope_m : HopeMode,
    /// describe range approximation mode
    range_m: RangeApproxMode,
    /// decay factor taking account number of hops away from a node
    decay_f : f64
} // 


impl HopeParams {

pub fn new(hope_m : HopeMode, range_m : RangeApproxMode, decay_f : f64) -> Self {
    HopeParams{hope_m, range_m, decay_f}
} // end of new 

pub fn get_hope_mode(&self) ->  HopeMode { self.hope_m}

pub fn get_decay_weight(&self) -> f64 {self.decay_f}

pub fn get_range_mode(&self) -> RangeApproxMode {self.range_m}

} // end of impl HopeParams


//============================================================



/// Structure for graph asymetric embedding with approximate random generalized svd to get an estimate of rank necessary
/// to get a required precision in the SVD. 
/// The structure stores the adjacency matrix in a full (ndarray) or compressed row storage format (using crate sprs).
//
pub struct Hope<F> {
    /// 
    params : HopeParams,
    /// the graph as a matrix
    mat : MatRepr<F>,
    ///
    _degrees : Option<Vec<Degree>>, 
    /// store the eigenvalue weighting the eigenvectors. This give information on precision.
    sigma_q : Option<Array1<F>>
}



impl <F> Hope<F>  where
    F: Float + Scalar + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + num_traits::MulAdd +
         Default + Send + Sync{

    pub fn new(params : HopeParams, trimat : TriMatI<F, usize>) -> Self {
        let csrmat = trimat.to_csr();
        let degrees = get_degrees(&csrmat);
        Hope::<F>{params, mat : MatRepr::from_csrmat(trimat.to_csr()), _degrees : Some(degrees), sigma_q : None}
    }

    /// instantiate a Hope problem with the adjacency matrix
    pub fn from_ndarray(params : HopeParams, mat : Array2<F>) -> Self {
        let mat = MatRepr::from_array2(mat);
        Hope::<F> {params, mat, _degrees: None, sigma_q : None}
    }

    pub fn get_nb_nodes(&self) -> usize {
        self.mat.shape()[0]
    }

    /// returns the quotients of eigenvalues.
    /// The relative precision of the embedding can be appreciated by the quantity quotient[quotient.len()-1]/quotient\[0\]
    pub fn get_quotient_eigenvalues(&self) -> Option<&Array1<F>> {
        match &self.sigma_q {
            Some(sigma) => {return Some(sigma); }
            _                                   => { return  None;}
        }
    } // end of get_quotient_eigenvalues

    // Noting A the adjacency matrix we constitute the couple (M_g, M_l ) = (I - β A, β A). 
    // We must check that beta is less than the spectral radius of adjacency matrix so that M_g is inversible.
    // In fact we go to the Gsvd with the pair (transpose(β A), transpose(I - β A))
    // as we search a representation of inverse(I - β P) * (β A)
    /// - factor helps defining the extent to which the neighbourhood of a node is taken into account when using the katz index matrix.
    ///     factor must be between 0. and 1.
    fn make_katz_problem(&self, factor : f64, approx_mode : RangeApproxMode) -> Result<GSvdApprox<F>, anyhow::Error> {
        //
        log::debug!("hope::make_katz_problem approx_mode : {:?}, factor : {:?}", approx_mode, factor);
        // enforce rule on factor
 /*        let radius = match self.mat.get_data() {
            MatMode::FULL(mat) =>  { estimate_spectral_radius_fullmat(&mat) },
            MatMode::CSR(csmat) =>  { estimate_spectral_radius_csmat(&csmat)},
        };   
        log::debug!("make katz_problem : got spectral radius : {}", radius);      */
        //  defining beta ensures that the matrix (Mg) in Hope paper is inversible.
        let radius = 1.;
        let beta = factor / radius;
        // now we can define a GSvdApprox problem
        // We must now define  A and B in Wei-Zhang paper or mat_g (global) and mat_l (local in Ou paper)
        // mat_g is beta * transpose(self.mat) but we must send it transpose to Gsvd  * transpose(self.mat)
        // For mat_l it is I - beta * &self.mat, but ust send transposed to Gsvd
        let mut mat_l = self.mat.transpose_owned();
        mat_l.scale(F::from_f64(beta).unwrap());
        // 
        let mat_g = compute_1_minus_beta_mat(&self.mat, beta, true);
/*         if log::log_enabled!(log::Level::Debug) {
            let new_radius = match mat_l.get_data() {
                MatMode::FULL(mat_l_full) =>  { estimate_spectral_radius_fullmat(&mat_l_full) },
                MatMode::CSR(csmat_l) =>  { estimate_spectral_radius_csmat(&csmat_l)},
            };
            log::debug!("make katz_problem : I - beta * A , got new spectral radius : {}", new_radius);     
        } */
        let gsvdapprox = GSvdApprox::new(mat_l, mat_g, approx_mode, None);
        //
        return Ok(gsvdapprox);
    } // end of make_katz_problem



    /// Noting A the adjacency matrix we constitute the couple (M_g, M_l ) = (I - β P, (1. - β) * I). 
    /// Has a good performance on link prediction Cf [https://dl.acm.org/doi/10.1145/3012704]
    /// A survey of link prediction in complex networks. Martinez, Berzal ACM computing Surveys 2016.
    /// 
    // In fact we go to the Gsvd with the pair (transpose((1. - β) * I)), transpose(I - β P))
    // as we search a representation of inverse(I - β P) * (1. - β) * I
    fn make_rooted_pagerank_problem(&mut self, factor : f64, approx_mode : RangeApproxMode) -> Result<GSvdApprox<F>, anyhow::Error> 
            where for<'r> F: std::ops::MulAssign<&'r F>  {
        //
        log::debug!("hope::make_rooted_pagerank_problem approx_mode : {:?}, factor : {:?}", approx_mode, factor);
        //
        crate::tools::renormalize::matrepr_row_normalization(& mut self.mat);
        // Mg is I - alfa * P where P is the normalized adjacency matrix to a probability matrix
        let mat_g = compute_1_minus_beta_mat(&self.mat, factor, true);
        // compute Ml = (1-alfa) I
        let mat_l = match self.mat.get_data() {
            MatMode::FULL(_) => { 
                let mut dense = Array2::<F>::eye(self.mat.shape()[0]);
                dense *= F::from_f64(1. - factor).unwrap();
                MatRepr::<F>::from_array2(dense)
            },
            MatMode::CSR(_) =>  {
                let mut id =  CsMat::<F>::eye(self.get_nb_nodes());
                id.scale(F::one() - F::from_f64(factor).unwrap());
                MatRepr::<F>::from_csrmat(id)
            },
        };
        let gsvdapprox = GSvdApprox::new(mat_l, mat_g , approx_mode, None);
        //
        return Ok(gsvdapprox);
    } // end of make_rooted_pagerank_problem


    fn embed_rpr_simple(&mut self, factor : f64, approx_mode : RangeApproxMode) -> Result<EmbeddedAsym<F>, anyhow::Error> 
            where for<'r> F: std::ops::MulAssign<&'r F>  {
        //
        log::debug!("hope::embed_rpr_simple : {:?}, factor : {:?}", approx_mode, factor);
        //
        crate::tools::renormalize::matrepr_row_normalization(& mut self.mat);
        // Mg is I - alfa * P where P is the normalized adjacency matrix to a probability matrix
        let t_mat_g = compute_1_minus_beta_mat(&self.mat, factor, true);
        // compute svd approx of transpose(mat_g) which U and V as inverse of Mg                 
        let mut svd_approx = SvdApprox::new(&t_mat_g);
        let svd_res = svd_approx.direct_svd(approx_mode); 
        if svd_res.is_err() {
            return Err(anyhow!("compute_embedded : ADA mode, call SvdApprox.direct_svd failed"));
        }
        let svd_res = svd_res.unwrap();
        // now we have svd approx de transpose(Matg_g) we have just to modify singular values
        let s = match svd_res.get_sigma() {
            Some(s) => { s },
                     _                       => { return  Err(anyhow!("embed_from_svd_result could not get s"));},
        };
        //
        let u : &Array2<F> = match svd_res.get_u() {
            Some(u) =>  { u },
                    _                        => { return  Err(anyhow!("compute_embedded could not get u")); },
        };
        let vt : &Array2<F> = match svd_res.get_vt() {
            Some(vt) =>  { vt },
                    _                        => { return  Err(anyhow!("compute_embedded could not get u")); },
        };
        //
        log::info!("nb eigen values {}, first eigenvalue {:.3e}, last eigenvalue : {:.3e}", s.len(), s[0], s[s.len()-1]);
        if log::log_enabled!(log::Level::Info) {
            for i in 0..20.min(s.len()-1) {
                log::debug!(" sigma_q i : {}, value : {:?} ", i, s[i]);
            }
        }
        //
        let nb_sigma = s.len();
        let mut source = Array2::<F>::zeros((self.get_nb_nodes(), nb_sigma));
        let mut target = Array2::<F>::zeros((self.get_nb_nodes(), nb_sigma)); 
        // source is U , target is V, we must inverse spectrum.
        let v = vt.t(); 
        assert_eq!(u.ncols(), v.ncols());
        //
        log::info!("nb eigen values {}, first eigenvalue {:.3e}, last eigenvalue : {:.3e}", s.len(), s[0], s[s.len()-1]);
        if log::log_enabled!(log::Level::Info) {
            for i in 0..20.min(s.len()-1) {
                log::debug!(" sigma_q i : {}, value : {:?} ", i, s[i]);
            }
        }
        log::trace!("setting embedding for nb_nodes : {}", self.get_nb_nodes());
        for i in 0..self.get_nb_nodes() {
            for j in 0..nb_sigma {
                assert!(s[j] > F::zero());
                let sigma = Float::sqrt(F::from(1.- factor).unwrap() /s[j]);
                source.row_mut(i)[j] = sigma * u.row(i)[j];
                target.row_mut(i)[j] = sigma * v.row(i)[j];
            }
            log::trace!("\n source {} {:?}", i, source.row(i));
            log::trace!("\n target {} {:?}", i, target.row(i));
        } 
        log::trace!("exiting embed_from_svd_result");
        let embedded_a = EmbeddedAsym::new(source, target, None, hope_distance);
        //
        return Ok(embedded_a);
    } // end of embed_rpr_simple



    // Noting A the adjacency matrix we constitute the couple (M_g, M_l ) = (I, adamic_ada transform of matrep) 
    // so we do not need Gsvd, a simple approximated svd is sufficient
    fn make_adamic_adar_problem(&mut self) -> Result<SvdApprox<F>, anyhow::Error>
             where F : Send + Sync + for<'r>  std::ops::MulAssign<&'r F>  {
        //

        log::debug!("hope::make_adamicadar_problem");
        crate::tools::renormalize::matrepr_adamic_adar_normalization(& mut self.mat);
        // Mg is I, so in fact it is useless we have a simple SVD to approximate
        let mat_l = &self.mat;
        let svd_approx = SvdApprox::new(mat_l);
        return Ok(svd_approx);
    } // end of make_rooted_pagerank_problem


    // fills in embedding from a gsvd
    fn embed_from_gsvd_result(&mut self, gsvd_res : &GSvdResult<F>) -> Result<EmbeddedAsym<F>,anyhow::Error> {
        // get k. How many eigenvalues for first matrix are 1. (The part in alpha before s1)
        let k = gsvd_res.get_k();
        log::debug!(" number (k) of eigenvalues of first matrix that are equal to 1. : {}", k);
        // if k > 0 {
        //     println!("k = {}, should be zero!", k);
        //     log::error!("hope::compute_embedded k = {}, should be zero!", k);
        //     std::process::exit(1);
        // }
        // Recall M_g is the first matrix, M_l the second of the Gsvd problem.
        // so we need to sort quotients of M_l/M_g eigenvalues i.e s2/s1
        let s1 : ArrayView1<F> = match gsvd_res.get_s1() {
            Some(s) =>  s,
            _ => { return  Err(anyhow!("compute_embedded could not get s1")); },
        };
        let s2 : ArrayView1<F> = match gsvd_res.get_s2() {
            Some(s) =>  s,
            _ => { return  Err(anyhow!("compute_embedded could not get s2")); },
        };
        let v1 : &Array2<F> = match gsvd_res.get_v1() {
            Some(s) =>  s,
            _ => { return  Err(anyhow!("compute_embedded could not get v1")); },
        };
        let v2 : &Array2<F> = match gsvd_res.get_v2() {
            Some(s) =>  s,
            _ => { return  Err(anyhow!("compute_embedded could not get v2")); },
        };
        //
        assert_eq!(s1.len() , s2.len());
        let mut sort_needed = false;
        // in theory sigma_q should be sorted in decreasing order
        let mut sigma_q = Vec::<IndexedValue<F>>::with_capacity(s1.len());
        for i in 0..s1.len() {
            log::debug!("s1 : {}, s2 : {}", s1[i], s2[i]);
            let last = sigma_q.last();
            if s1[i] > F::zero() {
                let new_val = IndexedValue::<F>(i,s2[i]/s1[i]);
                if last.is_some() {
                    if new_val.1 >= last.unwrap().1 {
                        sort_needed = true;
                        log::error!("non decreasing quotient of eigen values, must implement permutation");
                    }
                }
                sigma_q.push(new_val);
                log::debug!("i {} , s2/s1[i] {}", i , new_val.1);
            }
        }
        let mut permutation = Vec::<usize>::with_capacity(s1.len());
        if sort_needed {
            sigma_q.sort_unstable_by(decreasing_sort_nans_first);
        }
        for idx in &sigma_q {
            permutation.push(idx.0);
        }
        // Now we can construct Embedded
        // U_source (resp. U_target) corresponds to M_global (resp. M_local) i.e  first (resp. second) component of GsvdApprox
        //
        let nb_sigma = permutation.len();
        let mut source = Array2::<F>::zeros((self.get_nb_nodes(), nb_sigma));
        let mut target = Array2::<F>::zeros((self.get_nb_nodes(), nb_sigma));
        for i in 0..nb_sigma {
            let sigma = Float::sqrt(sigma_q[i].1);
            for j in 0..v1.ncols() {
                log::debug!(" sigma_q i : {}, value : {:?} ", i, sigma);
                source.row_mut(i)[j] = sigma * v1.row(permutation[i])[j];
                target.row_mut(i)[j] = sigma * v2.row(permutation[i])[j];
            }
        } 
        //
        if sigma_q.len() > 0 {
            log::info!("last eigen value to first : {}", sigma_q[sigma_q.len()-1].1/ sigma_q[0].1);
        }
        else {
            log::error!("compute_embedded : did not found eigenvalues in interval ]0., 1.[");
            return Err(anyhow!("compute_embedded : did not found eigenvalues in interval ]0., 1.["));
        }
        self.sigma_q = Some(Array1::from_iter(sigma_q.iter().map(|x| x.1)));
        //
        let embedded_a = EmbeddedAsym::new(source, target, None, hope_distance);
        //
        Ok(embedded_a)
    } // end of embed_from_gsvd_result


   // fills in embedding from a svd (and not a gsvd)! Covers the case Adamic Adar
   fn embed_ada_from_svd_result(&mut self, svd_res : &SvdResult<F>) -> Result<EmbeddedAsym<F>,anyhow::Error> {
        //
        log::debug!("entering embed_from_svd_result");
        //
        let s = match svd_res.get_sigma() {
            Some(s) => { s },
                     _                       => { return  Err(anyhow!("embed_from_svd_result could not get s"));},
        };
        //
        let u : &Array2<F> = match svd_res.get_u() {
            Some(u) =>  { u },
                    _                        => { return  Err(anyhow!("compute_embedded could not get u")); },
        };
        let vt : &Array2<F> = match svd_res.get_vt() {
            Some(vt) =>  { vt },
                    _                        => { return  Err(anyhow!("compute_embedded could not get u")); },
        };
        //
        let nb_sigma = s.len();
        let mut source = Array2::<F>::zeros((self.get_nb_nodes(), nb_sigma));
        let mut target = Array2::<F>::zeros((self.get_nb_nodes(), nb_sigma)); 
        let v = vt.t();  
        assert_eq!(u.ncols(), v.ncols());
        log::info!("nb eigen values {}, first eigenvalue {:.3e}, last eigenvalue : {:.3e}", s.len(), s[0], s[s.len()-1]);
        if log::log_enabled!(log::Level::Info) {
            for i in 0..20.min(s.len()-1) {
                log::debug!(" sigma_q i : {}, value : {:?} ", i, s[i]);
            }
        }
        log::trace!("setting embedding for nb_nodes : {}", self.get_nb_nodes());
        for i in 0..self.get_nb_nodes() {
            for j in 0..nb_sigma {
                let sigma = Float::sqrt(s[j]);
                source.row_mut(i)[j] = sigma * u.row(i)[j];
                target.row_mut(i)[j] = sigma * v.row(i)[j];
            }
            log::trace!("\n source {} {:?}", i, source.row(i));
            log::trace!("\n target {} {:?}", i, target.row(i));
        } 
        log::trace!("exiting embed_from_svd_result");
        let embedded_a = EmbeddedAsym::new(source, target, None, hope_distance);
        //
        return Ok(embedded_a);
   } // end of embed_from_svd_result


 

    /// computes the embedding
    /// - dampening_factor helps defining the extent to which the multi hop neighbourhood of a node is taken into account 
    ///   when using the katz index matrix or Rooted Page Rank. Factor must be between 0. and 1.
    /// 
    /// *Note that in RPR mode the matrix stored in the Hope structure is renormalized to a transition matrix!!*
    /// 
    pub fn compute_embedded(&mut self) -> Result<EmbeddedAsym<F>,anyhow::Error> {
        //
        log::debug!("hope::compute_embedded");
        let use_gsvd_for_rpr = false;
        //
        let cpu_start = ProcessTime::now();
        let sys_start = SystemTime::now();
        //
        let embedding = match self.params.hope_m {
            HopeMode::KATZ => {
                let gsvd_pb = self.make_katz_problem(self.params.get_decay_weight(), self.params.get_range_mode());
                let gsvd_res = gsvd_pb.unwrap().do_approx_gsvd(); 
                if gsvd_res.is_err() {
                    return Err(anyhow!("compute_embedded : KATZ mode, call GSvdApprox.do_approx_gsvd failed"));
                }
                let gsvd_res = gsvd_res.unwrap();
                let embedding = self.embed_from_gsvd_result(&gsvd_res);
                embedding
            },
            HopeMode::RPR => { 
                let embedding = match use_gsvd_for_rpr {      
                    true => {          
                        let gsvd_pb = self.make_rooted_pagerank_problem(self.params.get_decay_weight(), self.params.get_range_mode());
                        let gsvd_res = gsvd_pb.unwrap().do_approx_gsvd(); 
                        if gsvd_res.is_err() {
                            return Err(anyhow!("compute_embedded : RPR mode, call GSvdApprox.do_approx_gsvd failed"));
                        }
                        let gsvd_res = gsvd_res.unwrap();
                        let embedding = self.embed_from_gsvd_result(&gsvd_res);  
                        embedding  
                    },
                    false => {
                        log::debug!("trying RPR with simple svd");
                        let embedding = self.embed_rpr_simple(self.params.get_decay_weight(), self.params.get_range_mode());
                        embedding                        
                    },
                };
                embedding
            },
            HopeMode::ADA => {
                let range_mode = self.params.get_range_mode().clone();
                let svd_pb = self.make_adamic_adar_problem();
                let svd_res = svd_pb.unwrap().direct_svd(range_mode); 
                if svd_res.is_err() {
                    return Err(anyhow!("compute_embedded : ADA mode, call SvdApprox.direct_svd failed"));
                }
                let svd_res = svd_res.unwrap();
                let embedding = self.embed_ada_from_svd_result(&svd_res);
                embedding
            },
        };  // znd of match
        log::info!(" compute_embedded sys time(s) {:.2e} cpu time(s) {:.2e}", sys_start.elapsed().unwrap().as_secs(), cpu_start.elapsed().as_secs());
        //
        return embedding;
    }  // end of compute_embedded
}  // end of impl Hope



//====================================================================================


/// implement EmbedderT trait for Hope<F> where F is f64 or f32
impl <F> EmbedderT<F> for Hope<F>  
    where F : Float + Scalar + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + num_traits::MulAdd + 
        Default + Send + Sync {
    type Output = EmbeddedAsym<F>;
    ///
    fn embed(&mut self) -> Result<EmbeddedAsym<F>, anyhow::Error > {
        let res = self.compute_embedded();
        match res {
            Ok(embeded) => {
                return Ok(embeded);
            },
            Err(err) => { return Err(err);}
        }
    } // end of embed
} // end of impl<F> EmbedderT<F>



//                  Some utilities
// =================================================

 

// useful for Katz Index and Rooted Page Rank
// return Id -  β * mat if transpose == false or Id -  β * transpose(mat) if transpose == true
// cannot avoid allocations (See as Katz Index and Rooted Page Rank needs a reallocation for a different mat each! which
// forbid using reference in GSvdApprox if we want to keep one definition a GSvdApprox)
fn compute_1_minus_beta_mat<F>(mat : &MatRepr<F>, beta : f64, transpose : bool) -> MatRepr<F> 
        where  F : Sync + Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + Default {
    //
    match mat.get_data() {
        MatMode::FULL(mat) => {
            log::debug!("atp::hope compute_1_minus_beta_mat full case , beta : {:?}, transpose : {:?}", beta, transpose);
            let (nbrow, nbcol) = mat.dim();
            assert_eq!(nbrow, nbcol);
            let mut new_mat = ndarray::Array2::<F>::eye(nbrow);
            new_mat.scaled_add(- F::from_f64(beta).unwrap(), mat);          // BLAS axpy
            if transpose {
                return MatRepr::from_array2(new_mat.t().to_owned());
            }
            else {
                return MatRepr::from_array2(new_mat);
            }
        },
        // 
        MatMode::CSR(mat)  => { 
            log::debug!("atp::hope compute_1_minus_beta_mat csr case , beta : {:?}, transpose : {:?}", beta, transpose);
            assert_eq!(mat.rows(), mat.cols());
            let n = mat.rows();
            let nnz = mat.nnz();
            // get an iter on triplets, construct a new trimatb
            let mut rows = Vec::<usize>::with_capacity(nnz+ n);
            let mut cols = Vec::<usize>::with_capacity(nnz+n);
            let mut values = Vec::<F>::with_capacity(nnz+n);
            let mut iter = mat.iter();
            let beta_f = F::from_f64(beta).unwrap();
            while let Some((val, (row, col))) = iter.next() {
                if row != col {
                    if transpose {
                        rows.push(col);
                        cols.push(row);
                    }
                    else {
                        rows.push(row);
                        cols.push(col);                        
                    }
                    values.push(- beta_f * *val);
                } else {
                    log::info!("there was sthing i ({:?}, {:?}),  val {:?} ", row , col, val);
                }
            };
            // fill values in diag not already initialized
            for i in 0..n {
                    rows.push(i);
                    cols.push(i);
                    values.push(F::one()); 
            }
            let trimat = TriMatBase::<Vec<usize>, Vec<F>>::from_triplets((n,n),rows, cols, values);
            let csr_mat : CsMat<F> = trimat.to_csr();
            return MatRepr::from_csrmat(csr_mat);
        },
    }  // end of match
} // end of compute_1_minus_beta_mat




//========================================================================================


#[cfg(test)]
mod tests {

    //    RUST_LOG=graphembed::hope=DEBUG cargo test test_name -- --nocapture

use super::*;

use crate::io::csv::csv_to_trimat;

#[allow(unused)]
use annembed::tools::svdapprox::{RangePrecision, RangeRank};

use crate::prelude::*;


#[allow(dead_code)]
fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}  








#[test]
fn test_hope_gnutella09() {
    //
    log_init_test();
    log::info!("in hope::test_hope_gnutella09"); 
    // Nodes: 8114 Edges: 26013
    let path = std::path::Path::new(crate::DATADIR).join("p2p-Gnutella09.txt");
    log::info!("\n\n test_nodesketchasym_wiki, loading file {:?}", path);
    let res = csv_to_trimat::<f64>(&path, true, b'\t');
    if res.is_err() {
        log::error!("error : {:?}", res.as_ref().err());
        log::error!("hope::tests::test_hope_gnutella09 failed in csv_to_trimat");
        assert_eq!(1, 0);
    }
    let (trimat, node_index) = res.unwrap();
    let hope_m = HopeMode::KATZ;
    let decay_f = 0.05;
//    let range_m = RangeApproxMode::RANK(RangeRank::new(500, 2));
    let range_m = RangeApproxMode::EPSIL(RangePrecision::new(0.1, 10, 300));
    let params = HopeParams::new(hope_m, range_m, decay_f);
     // now we embed
    let mut hope = Hope::new(params, trimat); 
    let hope_embedding = Embedding::new(node_index, &mut hope);
    if hope_embedding.is_err() {
        log::error!("error : {:?}", hope_embedding.as_ref().err());
        log::error!("test_wiki failed in compute_Embedded");
        assert_eq!(1, 0);        
    }
    //    
    let _embed_res = hope_embedding.unwrap();

} // end of test_wiki



}  // end of mod test