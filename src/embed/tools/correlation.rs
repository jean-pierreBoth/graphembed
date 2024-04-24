//! Basic correlation

use num_traits::*;

use std::fmt::Debug;

/// spearman correlation coefficient
pub fn pearson_cor<'a, F>(v1: &'a [F], v2: &'a [F]) -> F
where
    F: Float + std::iter::Sum<&'a F> + FromPrimitive + Debug,
{
    assert_eq!(v1.len(), v2.len());
    // means...
    let mean1: F = v1.into_iter().sum::<F>() / F::from(v1.len()).unwrap();
    let mean2: F = v2.into_iter().sum::<F>() / F::from(v2.len()).unwrap();
    // variances
    let (mut s1, mut s2) =
        v1.into_iter()
            .zip(v2.into_iter())
            .fold((F::zero(), F::zero()), |acc, (t1, t2)| {
                (
                    acc.0 + (*t1 - mean1) * (*t1 - mean1),
                    acc.1 + (*t2 - mean2) * (*t2 - mean2),
                )
            });
    //
    s1 = (s1 / F::from(v1.len() - 1).unwrap()).sqrt();
    s2 = (s2 / F::from(v2.len() - 1).unwrap()).sqrt();
    // covariance...
    let cov = v1
        .into_iter()
        .zip(v2.into_iter())
        .fold(F::zero(), |acc, (t1, t2)| {
            acc + (*t1 - mean1) * (*t2 - mean2)
        })
        / F::from(v1.len() - 1).unwrap();
    //
    let rho = cov / (s1 * s2);
    if rho.abs() > F::one() {
        log::error!("correlation : {:?}", rho);
    }
    //
    rho
} // end of pearson_cor
