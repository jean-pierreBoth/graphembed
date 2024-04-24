//! compute degrees in and out from a csmat

use anyhow::anyhow;

use hdrhistogram::Histogram;
use indexmap::set::IndexSet;
use rand::distributions::{Uniform, WeightedIndex};
use sprs::CsMatI;
use std::collections::HashMap;

use rand::prelude::*;

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

//
/// samples (exactly) nb_nodes with weighs proportional to their degrees, return nodes rank
pub fn sample_nodes_by_degrees<F>(csmat: &CsMatI<F, usize>, nb_nodes: usize) -> Vec<usize>
where
    F: Copy + Default,
{
    if nb_nodes >= csmat.rows() {
        log::error!("graph has only {} nb_nodes ", csmat.rows());
    }
    let mut sampled = IndexSet::<usize>::with_capacity(nb_nodes);
    //
    let degrees = get_csmat_degrees(csmat);
    let weights: Vec<f32> = degrees
        .into_iter()
        .map(|d| (d.degree_in() + d.degree_out()) as f32)
        .collect();
    //
    let mut rng = thread_rng();
    let distribution = WeightedIndex::new(&weights).unwrap();
    let mut nb_try = 0;
    let mut nb_sampled = 0;
    //
    while nb_sampled < nb_nodes {
        let node = distribution.sample(&mut rng);
        nb_try += 1;
        if sampled.insert(node) {
            nb_sampled += 1;
        }
    }
    log::info!(
        "sample_nodes_by_degrees  nb_try : {} nb_sampled : {}",
        nb_try,
        nb_sampled
    );
    //
    let nodes = sampled.into_iter().map(|n| n).collect();
    nodes
} // end of sample_nodes_by_degrees

//
/// sample (approximately) nb_nodes uniformly
pub fn sample_nodes_uniform<F>(csmat: &CsMatI<F, usize>, nb_sample: usize) -> Vec<usize>
where
    F: Copy + Default,
{
    //
    assert_eq!(csmat.rows(), csmat.cols());
    //
    let nb_nodes = csmat.rows();
    assert!(nb_nodes <= i32::MAX as usize);
    //
    let uniform = Uniform::<f64>::new(0., 1.);
    let mut rng = thread_rng();
    //
    let fraction = nb_sample as f64 / nb_nodes as f64;
    let sampled_nodes: Vec<usize> = (0..nb_nodes)
        .into_iter()
        .map(|i| {
            if uniform.sample(&mut rng) <= fraction {
                i as i32
            } else {
                -1
            }
        })
        .filter(|i| *i >= 0)
        .map(|i| i as usize)
        .collect();
    //
    sampled_nodes
} // end of sample_nodes_uniform
