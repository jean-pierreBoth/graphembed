//! some utilities related to matrix renormalization
//! -  go from adjacency matrix to probability transition matrix
//! - from adjacency matrix to adamic adar normalization

use anyhow::{anyhow};

use ndarray_linalg::{Scalar, Lapack};
use num_traits::float::Float;

use ndarray::{Array1, Array2, ArrayView2};
use sprs::{CsMat, TriMatBase};

use annembed::tools::svdapprox::*;


/// do a Row normalization of Csr Mat, to get a transition matrix from an adjacency matrix.
pub fn csr_row_normalization<F>(csr_mat : &mut CsMat<F>) where 
            F : Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F>  {
    //
    assert!(csr_mat.is_csr());
    //
    let (nb_row, _ ) = csr_mat.shape();
    let mut range_i : std::ops::Range<usize>;
    let mut nb_null_row = 0usize;
    for i in 0..nb_row {
        let mut sum_i : F;
        {  // the borrow checker do not let us access csr_mat.indptr() and csr_mat.data_mut() simultaneously
            range_i = csr_mat.indptr().outer_inds_sz(i);
            sum_i = F::zero();
            let data = csr_mat.data();
            for j in range_i.clone() {
                sum_i = sum_i + data[j];
            }
        }   
        // we sum of row i
        if !(sum_i > F::zero()) {
            log::trace!("csr_row_normalization null sum of row i {}", i);
            nb_null_row += 1;
        } else {
            let data = csr_mat.data_mut();
            for j in range_i {
                data[j] = data[j]/sum_i;
            }
        }
    } // end of for i
    log::debug!("csr_row_normalization nb row with null sum : {}", nb_null_row);
} // end of csr_row_normalization


/// do a Row normalization of Csr Mat, to get a transition matrix from an adjacency matrix.
pub fn dense_row_normalization<F>(mat : &mut Array2<F>) 
    where  F : Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> {
    //
    let (nb_row, _) = mat.dim();
    let mut nb_null_row = 0usize;
    for i in 0..nb_row {
        let mut row = mat.row_mut(i);
        let sum_i = row.sum();
        if!(sum_i > F::zero()) {
            row.map_mut(|x| *x = *x/sum_i);
        }
        else {
            nb_null_row += 1;
            log::trace!("dense_row_normalization null sum of row i {}", i);
        }
    }
    log::debug!("dense_row_normalization nb row with null sum : {}", nb_null_row);
}  // end of for dense_row_normalization



// do a Row normalization for dense or csr matrix
pub fn matrepr_row_normalization<F> (mat : &mut MatRepr<F>)
    where  F : Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + Default {
    match &mut mat.get_data_mut() {
        MatMode::FULL(mat) => { dense_row_normalization(mat) }, 
        MatMode::CSR(csrmat) =>  { 
                    assert!(csrmat.is_csr());
                    csr_row_normalization(csrmat) }
    }
} // end of matrepr_row_normalization


/// do adamic adar renormalization for full matrix and return modified matrix.
/// Matrix must be square
pub fn adamic_adar_normalization_full<F> (mat : &mut Array2<F>) -> Result<(), anyhow::Error>
    where  F : Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + Default
{
    let (nb_row, nb_col) = mat.dim();
    assert_eq!(nb_row, nb_col);
    //
    let mut diagonal = Array1::<F>::zeros(nb_row);
    for i in 0..nb_row {
        diagonal[i] += mat.row(i).sum();
    }
    let tmat = mat.t();
    for i in 0..nb_row {
        diagonal[i] += tmat.row(i).sum();
    }
    // 
    for i in 0..nb_row {
        if diagonal[i] <= F::zero() {
            log::error!("adamic_adar_normalization_full , cannot do adamic transform,  null coeff at index  {}", i);
            return Err(anyhow!("adamic_adar_normalization_full , cannot do adamic transform,  null coeff at index  {}", i));
        }
        // TODO possibly we can set term to 0! as it means a node is isolated and cannot appear in multplication of matrix
        diagonal[i] = F::one()/diagonal[i];
    }
    // compute Diag*mat
    let mut adamat = mat.clone().to_owned(); 
    for i in 0..nb_row {
        adamat.row_mut(i).map_inplace( |x|  { *x = *x * diagonal[i];});
    }
    // compute mat * (adamat) = mat * (diagonal * mat)
    *mat = mat.dot(&adamat);
    //
    return Ok(());
}  // end of adamic_adar_normalization


// return a Csmat with adamic adar renormalization
pub fn adamic_adar_normalization_csmat<F> (mat : &mut CsMat<F>) -> Result<(), anyhow::Error>
    where  F : Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + Default
{
    let (nb_row, nb_col) = mat.shape();
    assert_eq!(nb_row, nb_col);
    //
    let nnz = mat.nnz();
    let mut diagonal = Array1::<F>::zeros(nb_row);
    // allocate triplets for adamic adamar transform
    let mut rows : Vec<usize> = Vec::<usize>::with_capacity(2*nnz);
    let mut cols : Vec<usize> = Vec::<usize>::with_capacity(2*nnz);
    let mut values : Vec<F> = Vec::<F>::with_capacity(2*nnz);
    //
    let mut cs_iter = mat.iter();
    while let Some(item) = cs_iter.next() {
        let row = item.1.0;
        let col = item.1.1;
        diagonal[row] += *item.0;
        diagonal[col] += *item.0;
        rows.push(row);
        cols.push(col);
        values.push(*item.0);
    }
    // fill in TriMatBase corresponding to DA
    for i in 0..values.len() {
        let row = rows[i];
        values[i] = values[i] / diagonal[row];
    }
    let tri_da = TriMatBase::from_triplets((nb_row, nb_col), rows, cols, values);
    // now we have compute D*mat in tri_da, must compute mat * tri_da to get adar in trimat format a.k.a tri_adar

    //
    return Err(anyhow!("not yet implemented"));
}  // end of adamic_adar_normalization_csmat

//===============================================================


mod tests {

    #[allow(unused)]
    use super::*;

    use sprs::{CsMat, TriMatBase};

    #[allow(dead_code)]
    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }


    #[allow(unused)]
    fn get_wiki_csr_mat_f64() -> CsMat<f64> {
        // 
        // let mat =  ndarray::arr2( & 
        //   [[ 1. , 0. , 0. , 0., 2. ],  // row 0
        //   [ 0. , 0. , 3. , 0. , 0. ],  // row 1
        //   [ 0. , 1. , 0. , 0. , 0. ],  // row 2
        //   [ 0. , 2. , 0. , 4. , 0. ]]  // row 3
        // );
        let mut rows = Vec::<usize>::with_capacity(5);
        let mut cols = Vec::<usize>::with_capacity(5);
        let mut values = Vec::<f64>::with_capacity(5);
        rows.push(0);
        cols.push(0);
        values.push(1.);
        rows.push(0);
        cols.push(4);
        values.push(2.);
        // row 1    
        rows.push(1);
        cols.push(2);
        values.push(3.); 
        // row 2
        rows.push(2);
        cols.push(1);
        values.push(1.);         
        // row 3
        rows.push(3);
        cols.push(1);
        values.push(2.);
        rows.push(3);
        cols.push(3);
        values.push(4.);             
        //
        let trimat = TriMatBase::<Vec<usize>, Vec<f64>>::from_triplets((4,5),rows, cols, values);
        let csr_mat : CsMat<f64> = trimat.to_csr();
        csr_mat
    }  // end of get_wiki_csr_mat_f64
    


#[test]
    fn test_csr_row_normalization() {
        //
        log_init_test();
        //
        let mut csr_mat = get_wiki_csr_mat_f64();
        csr_row_normalization(&mut csr_mat);
        //
        let dense = csr_mat.to_dense();
        let check = (dense[[0,0]]- 1./3.).abs();
        log::debug!("check (0,0): {}", check);
        assert!( check < 1.0E-10);
        //
        let check = (dense[[0,4]]- 2./3.).abs();
        log::debug!("check (0,4): {}", check);
        assert!( check < 1.0E-10); 
        //       
        let check = (dense[[3,1]]- 1./3.).abs();
        log::debug!("check (3,1): {}", check);
        assert!( check < 1.0E-10);
   } // end of test_csr_row_normalization


   #[test]
    fn test_full_row_normalization() {
        //
        log_init_test(); 
        // 
        let mut dense =  ndarray::arr2( & 
          [[ 1. , 0. , 0. , 0., 2. ],  // row 0
          [ 0. , 0. , 3. , 0. , 0. ],  // row 1
          [ 0. , 1. , 0. , 0. , 0. ],  // row 2
          [ 0. , 2. , 0. , 4. , 0. ]]  // row 3
        );
        dense_row_normalization(&mut dense);
        let check = num_traits::Float::abs(dense[[0,0]]- 1./3.);
        log::debug!("check (0,0): {}", check);
        assert!( check < 1.0E-10);
        //
        let check = num_traits::Float::abs(dense[[0,4]]- 2./3.);
        log::debug!("check (0,4): {}", check);
        assert!( check < 1.0E-10); 
        //       
        let check = num_traits::Float::abs(dense[[3,1]]- 1./3.);
        log::debug!("check (3,1): {}", check);
        assert!( check < 1.0E-10);
    }


}   // end of mod tests