//! compute degrees in and out from a csmat

use anyhow::anyhow;

use hdrhistogram::Histogram;
use sprs::CsMatI;
use std::collections::HashMap;

/// first component is in, second component is out!
#[derive(Copy, Clone, Debug)]
pub struct Degree {
    pub d_in: u32,
    pub d_out: u32,
}

impl Degree {
    fn new(d_in: u32, d_out: u32) -> Self {
        Degree { d_in, d_out }
    }
    /// get degrre in
    pub fn degree_in(&self) -> u32 {
        self.d_in
    }
    /// get degree out
    pub fn degree_out(&self) -> u32 {
        self.d_out
    }
} // end of impl Degree

/// returns a vector of 2-uple consisting of degrees (in, out)
/// fist component is in, second component is out!
/// Self loops are not taken into account as the objective of this function
/// to be able to delete edge in AUC link prediction and avoiding making disconnected nodes
pub(crate) fn get_csmat_degrees<F>(csmat: &CsMatI<F, usize>) -> Vec<Degree>
where
    F: Copy + Default,
{
    //
    assert!(csmat.is_csr());
    //
    let (nb_row, _) = csmat.shape();
    let mut degrees = (0..nb_row)
        .into_iter()
        .map(|_| Degree::new(0, 0))
        .collect::<Vec<Degree>>();
    //
    let mut iter = csmat.iter();
    while let Some((_val, (i, j))) = iter.next() {
        if i != j {
            degrees[i].d_out += 1; // one more out for i
            degrees[j].d_in += 1; // one more in for j
        }
    }
    degrees
} // end of get_degrees

//
/// return degree histogram from trimat representation
pub fn get_degree_quant_from_trimat<F>(
    csmat: &CsMatI<F, usize>,
    symetric: bool,
) -> anyhow::Result<Histogram<u32>> {
    //
    let res_histo = Histogram::<u32>::new(3);
    if res_histo.is_err() {
        log::error!("histogram creation failed");
        return Err(anyhow!("histogram creation failed"));
    }
    let mut histo = res_histo.unwrap();
    let nb_rows = csmat.rows();
    let nb_cols = csmat.cols();
    let mut degrees = HashMap::<usize, u32>::with_capacity(nb_rows.max(nb_cols));
    //
    let mut mat_iter = csmat.iter();
    while let Some((_d, (i, j))) = mat_iter.next() {
        let i_opt = degrees.get_mut(&i);
        match i_opt {
            None => {
                degrees.insert(i, 1u32);
            }
            Some(count) => {
                *count += 1;
            }
        };
        if !symetric {
            let j_opt = degrees.get_mut(&j);
            match j_opt {
                None => {
                    degrees.insert(j, 1u32);
                }
                Some(count) => {
                    *count += 1;
                }
            };
        }
    }
    //
    for (_, v) in &degrees {
        histo.record(*v as u64).unwrap();
    }
    //
    return Ok(histo);
}

//
/// get degrees quantiles from a csr mat
pub fn get_degree_quant_from_csrmat<F>(csmat: &CsMatI<F, usize>) -> anyhow::Result<Histogram<u32>>
where
    F: Copy + Default,
{
    let degrees = get_csmat_degrees(csmat);
    let res_histo = Histogram::<u32>::new(3);
    if res_histo.is_err() {
        log::error!("histogram creation failed");
        return Err(anyhow!("histogram creation failed"));
    }
    let mut degree_histogram = res_histo.unwrap();
    for d in &degrees {
        degree_histogram.record(d.d_in.into()).unwrap();
    }
    //
    let nbslot = 20;
    let mut qs = Vec::<f64>::with_capacity(30);
    for i in 1..nbslot {
        let q = i as f64 / nbslot as f64;
        qs.push(q);
    }
    qs.push(0.99);
    qs.push(0.999);
    for q in qs {
        log::info!(
            "fraction : {:.3e}, degree : {}",
            q,
            degree_histogram.value_at_quantile(q)
        );
    }
    //
    Ok(degree_histogram)
}
