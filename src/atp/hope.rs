//! Asymetric Graph Embedding
//! Based on the paper:
//!     Asymetric Transitivity Preserving Graph Embedding 
//!     Ou, Cui Pei, Zhang, Zhu   in KDD 2016
//!  See  [atp](https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf)
//! 


use anyhow::{anyhow};

use ndarray::{Array1, Array2};

use ndarray_linalg::{Scalar, Lapack};



use num_traits::float::*; 
// use num_traits::cast::FromPrimitive;

use sprs::prod;
use sprs::{CsMat, TriMatBase};


use annembed::tools::svdapprox::{MatRepr, MatMode, RangeApproxMode};

use super::randgsvd::{GSvdApprox};
use super::orderingf::*;
use crate::embedding::EmbeddingAsym;


///
/// To specify if we run with Katz index or in Rooted Page Rank 
pub enum HopeMode {
    /// Katz index mode
    KATZ,
    /// Rooted Page Rank
    RPR,
} // end of HopeMode



/// Structure for graph asymetric embedding with approximate random generalized svd to get an estimate of rank necessary
/// to get a required precision in the SVD. 
/// The structure stores the adjacency matrix in a full (ndarray) or compressed row storage format (using crate sprs).
pub struct Hope<F> {
    /// the graph as a matrix
    mat : MatRepr<F>,
    /// store the eigenvalue weighting the eigenvectors. This give information on precision.
    sigma_q : Option<Array1<F>>
}



impl <F> Hope<F>  where
    F: Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + Default {

    /// instantiate a Hope problem with the adjacency matrix
    pub fn from_ndarray(mat : Array2<F>) -> Self {
        let mat = MatRepr::from_array2(mat);
        Hope::<F> {mat, sigma_q : None}
    }

    pub fn get_nb_nodes(&self) -> usize {
        self.mat.shape()[0]
    }

    /// returns the quotients of eigenvalues.
    /// The relative precision of the embedding can be appreciated by the quantity quotient[quotient.len()-1]/quotient[0]
    pub fn get_quotient_eigenvalues(&self) -> Option<&Array1<F>> {
        match &self.sigma_q {
            Some(sigma) => {return Some(sigma); }
            _                                   => { return  None;}
        }
    } // end of get_quotient_eigenvalues

    // Noting A the adjacency matrix we constitute the couple (M_g, M_l ) = (I - β A, β A). 
    // We must check that beta is less than the spectral radius of adjacency matrix so that M_g is inversible.
    // In fact we go to the Gsvd with the pair (transpose(β A), transpose(I - β A))

    /// - factor helps defining the extent to which the neighbourhood of a node is taken into account when using the katz index matrix.
    ///     factor must be between 0. and 1.
    fn make_katz_problem(&self, factor : f64, approx_mode : RangeApproxMode) -> GSvdApprox<F> {
        // enforce rule on factor
        let radius = match self.mat.get_data() {
            MatMode::FULL(mat) =>  { estimate_spectral_radius_fullmat(&mat) },
            MatMode::CSR(csmat) =>  { estimate_spectral_radius_csmat(&csmat)},
        };        
        //  defining beta ensures that the matrix (Mg) in Hope paper is inversible.
        let beta = factor / radius;
        // now we can define a GSvdApprox problem
        // We must now define  A and B in Wei-Zhang paper or mat_g (global) and mat_l (local in Ou paper)
        // mat_g is beta * transpose(self.mat) but we must send it transpose to Gsvd  * transpose(self.mat)
        // For mat_l it is I - beta * &self.mat, but ust send transposed to Gsvd
        let mut mat_g = self.mat.transpose_owned();
        mat_g.scale(F::from_f64(beta).unwrap());
        // 
        let mat_l = compute_1_minus_beta_mat(&self.mat, beta, true);
        let gsvdapprox = GSvdApprox::new(mat_g, mat_l, approx_mode, None);
        //
        return gsvdapprox;
    } // end of make_katz_pair



    /// Noting A the adjacency matrix we constitute the couple (M_g, M_l ) = (I - β P, (1. - β) * I). 
    /// Has a good performance on link prediction Cf [https://dl.acm.org/doi/10.1145/3012704]
    /// A survey of link prediction in complex networks. Martinez, Berzal ACM computing Surveys 2016.
    // In fact we go to the Gsvd with the pair (transpose((1. - β) * I)), transpose(I - β P))
    fn make_rooted_pagerank_problem(&mut self, factor : f64, approx_mode : RangeApproxMode) -> GSvdApprox<F> where
                    for<'r> F: std::ops::MulAssign<&'r F>  {
        //
        crate::renormalize::matrepr_row_normalization(& mut self.mat);
        // Mg is I - alfa * P where P is normalizez adjacency matrix to a probability matrix
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
                }
        };
        let gsvdapprox = GSvdApprox::new(mat_g, mat_l , approx_mode, None);
        //
        return gsvdapprox;
    } // end of make_rooted_pagerank_problem



    /// computes the embedding.
    /// - dampening_factor helps defining the extent to which the multi hop neighbourhood of a node is taken into account 
    ///   when using the katz index matrix or Rooted Page Rank. Factor must be between 0. and 1.
    /// 
    /// *Note that in RPR mode the matrix stored in the Hope structure is renormalized to a transition matrix!!*
    /// 
    pub fn compute_embedding(&mut self, mode :HopeMode, approx_mode : RangeApproxMode, dampening_f : f64) -> Result<EmbeddingAsym<F>,anyhow::Error> {
        //
        let gsvd_pb = match mode {
            HopeMode::KATZ => { self.make_katz_problem(dampening_f, approx_mode) },
            HopeMode::RPR => { self.make_rooted_pagerank_problem(dampening_f, approx_mode) },
        };
        // now we can approximately solve svd problem
        let gsvd_res = gsvd_pb.do_approx_gsvd(); 
        if gsvd_res.is_err() {
            return Err(anyhow!("compute_embedding : call GSvdApprox.do_approx_gsvd failed"));
        }
        let gsvd_res = gsvd_res.unwrap();
        // Recall M_g is the first matrix, M_l the second of the Gsvd problem.
        // so we need to sort quotients of M_l/M_g eigenvalues i.e s2/s1
        let s1 : &Array1<F> = match gsvd_res.get_s1() {
            Some(s) =>  s,
            _ => { return  Err(anyhow!("compute_embedding could not get s1")); },
        };
        let s2 : &Array1<F> = match gsvd_res.get_s2() {
            Some(s) =>  s,
            _ => { return  Err(anyhow!("compute_embedding could not get s2")); },
        };
        let v1 : &Array2<F> = match gsvd_res.get_v1() {
            Some(s) =>  s,
            _ => { return  Err(anyhow!("compute_embedding could not get v1")); },
        };
        let v2 : &Array2<F> = match gsvd_res.get_v2() {
            Some(s) =>  s,
            _ => { return  Err(anyhow!("compute_embedding could not get v2")); },
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
        // Now we can construct embedding
        // U_source (resp. U_target) corresponds to M_global (resp. M_local) i.e  first (resp. second) component of GsvdApprox
        //
        let nb_sigma = permutation.len();
        let mut source = Array2::<F>::zeros((self.get_nb_nodes(), nb_sigma));
        let mut target = Array2::<F>::zeros((self.get_nb_nodes(), nb_sigma));
        for i in 0..nb_sigma {
            let sigma = sigma_q[i].1;
            for j in 0..v1.ncols() {
                log::debug!(" sigma_q i : {}, value : {:?} ", i, sigma);
                source.row_mut(i)[j] = sigma * v1.row(permutation[i])[j];
                target.row_mut(i)[j] = sigma * v2.row(permutation[i])[j];
            }
        } 
        //
        log::info!("last eigen value to first : {}", sigma_q[sigma_q.len()-1].1/ sigma_q[0].1);
        self.sigma_q = Some(Array1::from_iter(sigma_q.iter().map(|x| x.1)));
        //
        let embeddinga = EmbeddingAsym::new(source, target);
        //
        Ok(embeddinga)
    }  // end of compute_embedding
}  // end of impl Hope



   // iterate in positive unit norm vector. mat is compressed row matrix and has all coefficients positive
fn estimate_spectral_radius_csmat<F>(mat : &CsMat<F>) -> f64 
        where F : Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc  {
        //
    let dims = mat.shape();
    let init = F::from_f64(1./(dims.0 as f64).sqrt()).unwrap();
    let mut v1 = Array1::<F>::from_elem(dims.0, init);
    let mut v2 = Array1::<F>::from_elem(dims.0, F::zero());
    let mut iter = 0;
    let mut radius: F;
    let epsil = F::from_f64(1.0E-5).unwrap();
    loop {
        v2.fill(F::zero());
        let v2_slice = v2.as_slice_mut().unwrap();
        prod::mul_acc_mat_vec_csr(mat.view(), v1.as_slice().unwrap(), v2_slice);
        radius = Scalar::sqrt(v2.dot(&v2));
        v2 = v2 * F::one()/ radius;
        let w = &v1 - &v2;
        let delta = Scalar::sqrt(w.dot(&w));
        iter += 1;
        if iter >= 1000 || delta < epsil {
            log::info!(" estimated radius at iter {} {}", iter, radius.to_f64().unwrap());
            break;
        }
        v1.assign(&v2);
    }
    return radius.to_f64().unwrap();
}   // end of estimate_spectral_radius_csmat



fn estimate_spectral_radius_fullmat<F>(mat : &Array2<F>) -> f64 
        where F : Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc {
    let dims = mat.dim();
    let init = F::from_f64(1./(dims.0 as f64).sqrt()).unwrap();
    let mut v1 = Array1::<F>::from_elem(dims.0, init);
    let mut v2: Array1::<F>;
    let mut iter = 0;
    let epsil = F::from_f64(1.0E-5).unwrap();
    let mut radius : F;
    loop {
        v2 = mat.dot(&v1);
        radius = Scalar::sqrt(v2.dot(&v2));
        v2 = v2 * F::one()/ radius;
        let w = &v1 - &v2;
        let delta = Scalar::sqrt(w.dot(&w));
        iter += 1;
        if iter >= 1000 || delta < epsil {
            log::info!(" estimated radius at iter {} {}", iter, radius.to_f64().unwrap());
            break;
        }
        v1 = v2;
    }
    //
    return radius.to_f64().unwrap();
} // end of estimate_spectral_radius_fullmat


// useful for Katz Index and Rooted Page Rank
// return Id -  β * mat if transpose == false or Id -  β * transpose(mat) if transpose == true
// cannot avoid allocations (See as Katz Index and Rooted Page Rank needs a reallocation for a different mat each! which
// forbid using reference in GSvdApprox if we want to keep one definition a GSvdApprox)
fn compute_1_minus_beta_mat<F>(mat : &MatRepr<F>, beta : f64, transpose : bool) -> MatRepr<F> 
        where  F : Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + Default {
    //
    match mat.get_data() {
        MatMode::FULL(mat) => {
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
            log::trace!("compute_1_minus_beta_mat");
            assert_eq!(mat.rows(), mat.cols());
            let n = mat.rows();
            let nnz = mat.nnz();
            // get an iter on triplets, construct a new trimatb
            let mut rows = Vec::<usize>::with_capacity(nnz+ n);
            let mut cols = Vec::<usize>::with_capacity(nnz+n);
            let mut values = Vec::<F>::with_capacity(nnz+n);
            let mut already_diag = Array1::<u8>::zeros(n);
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
                    values.push(F::one() - beta_f * *val);
                    already_diag[row] = 1;
                }
            };
            // fill values in diag not already initialized
            for i in 0..n {
                if already_diag[i] == 0 {
                    rows.push(i);
                    cols.push(i);
                    values.push(F::one()); 
                }                   
            }
            let trimat = TriMatBase::<Vec<usize>, Vec<F>>::from_triplets((4,5),rows, cols, values);
            let csr_mat : CsMat<F> = trimat.to_csr();
            return MatRepr::from_csrmat(csr_mat);
        },
    }  // end of match
} // end of compute_1_minus_beta_mat




//========================================================================================


mod tests {

#[allow(unused)]
use super::*;

#[allow(unused)]
use sprs::{CsMat, TriMatBase};

#[allow(dead_code)]
fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}  

// example from https://en.wikipedia.org/wiki/Spectral_radius
#[test]
fn test_spectral_radius_full() {
    log_init_test();
    log::info!("in test_spectral_radius_full");
    //
    let mat = ndarray::arr2(&[ [9., -1., 2.],
                                                                [-2., 8., 4.],
                                                                [1., 1., 8.]  ]);
    let radius = estimate_spectral_radius_fullmat(&mat);
    log::info!("estimate_spectral_radius_fullmat radius : {}", radius);
    //
    assert!((radius-10.).abs() < 1.0E-4);
} // end of test_spectral_radius_full


#[test]
fn test_spectral_radius_csr() {
    log_init_test();
    log::info!("in test_spectral_radius_csr"); 
    // compute radius from a full matrix
    let mat = ndarray::arr2(&[ [9., 0. , 2.],
        [0., 8., 4.],
        [1., 1., 0.]  ]);
        
    // check we get the same radius from the same matrix givn as a csr
    let mut rows = Vec::<usize>::with_capacity(6);
    let mut cols = Vec::<usize>::with_capacity(6);
    let mut values = Vec::<f64>::with_capacity(6);
    // row 0
    rows.push(0);
    cols.push(0);
    values.push(9.);

    rows.push(0);
    cols.push(2);
    values.push(2.);
    // row 1    
    rows.push(1);
    cols.push(1);
    values.push(8.);

    rows.push(1);
    cols.push(2);
    values.push(4.); 
    // row 2
    rows.push(2);
    cols.push(0);
    values.push(1.);

    rows.push(2);
    cols.push(1);
    values.push(1.);
    //
    let trimat = TriMatBase::<Vec<usize>, Vec<f64>>::from_triplets((3,3),rows, cols, values);
    let csr_mat : CsMat<f64> = trimat.to_csr();
    //
    let radius_from_full = estimate_spectral_radius_fullmat(&mat);
    log::info!("estimate_spectral_radius_fullmat radius : {}", radius_from_full);
    let radius_from_csmat = estimate_spectral_radius_csmat(&csr_mat);
    log::info!("estimate_spectral_radius_csmat radius : {}", radius_from_csmat);
    //
    assert!((radius_from_full-radius_from_csmat).abs() < 0.0001 * radius_from_full);


}  // enf of test_spectral_radius_csr



}  // end of mod test