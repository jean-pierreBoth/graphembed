//! Python bindings for graphembed  (pyo3 0.21 tested)
#![cfg(feature = "python")]

use std::path::Path;

use anyhow::{anyhow, Result};
use indexmap::IndexSet;
use log::info;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;         // Python, Bound<'_ ,PyModule>, etc.
use pyo3::wrap_pyfunction;
use sprs::TriMatI;

/* ---------- re‑exports from the graphembed crate ------------------------ */

use crate::prelude::{
    bson_dump, csv_to_trimat_delimiters, link, Embedding, Hope, HopeMode,
    HopeParams, NodeSketch, NodeSketchAsym, NodeSketchParams, RangeApproxMode,
    RangePrecision, RangeRank,
};
use crate::embedding::EmbedderT;
/* ----------------------------------------------------------------------- */
/* Helpers                                                                 */
/* ----------------------------------------------------------------------- */

fn to_py_err(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{:#}", err))
}

fn load_csv(
    csv: &str,
    symetric: bool,
) -> Result<(TriMatI<f64, usize>, IndexSet<usize>)> {
    let path = Path::new(csv);
    if !path.exists() {
        return Err(anyhow!("CSV file {:?} not found", csv));
    }
    // second boolean ==  “is asym file?” in original helper
    csv_to_trimat_delimiters::<f64>(path, !symetric)
}

/* ----------------------------------------------------------------------- */
/* EMBEDDING                                                               */
/* ----------------------------------------------------------------------- */

#[pyfunction]
#[pyo3(signature = (csv, target_rank, nbiter, symetric=true, output=None))]
fn embed_hope_rank(
    csv: &str,
    target_rank: usize,
    nbiter: usize,
    symetric: bool,
    output: Option<String>,
) -> PyResult<()> {
    let (trimat, nodes) = load_csv(csv, symetric).map_err(to_py_err)?;
    let params = HopeParams::new(
        HopeMode::ADA,
        RangeApproxMode::RANK(RangeRank::new(target_rank, nbiter)),
        1.0,
    );
    let mut hope = Hope::new(params, trimat);
    let emb = Embedding::new(nodes, &mut hope).map_err(to_py_err)?;

    if let Some(name) = output {
        let out = crate::io::output::Output::new(
            crate::io::output::Format::BSON,
            true,
            &Some(name),
        );
        bson_dump(&emb, &out).map_err(to_py_err)?;
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (csv, epsil, maxrank, blockiter, symetric=true, output=None))]
fn embed_hope_precision(
    csv: &str,
    epsil: f64,
    maxrank: usize,
    blockiter: usize,
    symetric: bool,
    output: Option<String>,
) -> PyResult<()> {
    let (trimat, nodes) = load_csv(csv, symetric).map_err(to_py_err)?;
    let params = HopeParams::new(
        HopeMode::ADA,
        RangeApproxMode::EPSIL(RangePrecision::new(epsil, blockiter, maxrank)),
        1.0,
    );
    let mut hope = Hope::new(params, trimat);
    let emb = Embedding::new(nodes, &mut hope).map_err(to_py_err)?;

    if let Some(name) = output {
        let out = crate::io::output::Output::new(
            crate::io::output::Format::BSON,
            true,
            &Some(name),
        );
        bson_dump(&emb, &out).map_err(to_py_err)?;
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (csv, decay, dim, nbiter, symetric=true, output=None))]
fn embed_sketching(
    csv: &str,
    decay: f64,
    dim: usize,
    nbiter: usize,
    symetric: bool,
    output: Option<String>,
) -> PyResult<()> {
    let (trimat, nodes) = load_csv(csv, symetric).map_err(to_py_err)?;
    let params = NodeSketchParams {
        sketch_size: dim,
        decay,
        nb_iter: nbiter,
        symetric,
        parallel: true,
    };

    if params.symetric {
        let mut s = NodeSketch::new(params, trimat);
        let emb = Embedding::new(nodes, &mut s).map_err(to_py_err)?;
        if let Some(name) = &output {
            let out = crate::io::output::Output::new(
                crate::io::output::Format::BSON,
                true,
                &Some(name.clone()),
            );
            bson_dump(&emb, &out).map_err(to_py_err)?;
        }
    } else {
        let mut s = NodeSketchAsym::new(params, trimat);
        let emb = Embedding::new(nodes, &mut s).map_err(to_py_err)?;
        if let Some(name) = &output {
            let out = crate::io::output::Output::new(
                crate::io::output::Format::BSON,
                true,
                &Some(name.clone()),
            );
            bson_dump(&emb, &out).map_err(to_py_err)?;
        }
    }
    Ok(())
}

/* ----------------------------------------------------------------------- */
/* VALIDATION  (return Vec<f64> like Rust API)                              */
/* ----------------------------------------------------------------------- */

#[pyfunction]
#[pyo3(signature = (csv, target_rank, nbiter, nbpass=1, skip_frac=0.2, symetric=true, centric=false))]
fn validate_hope_rank(
    csv: &str,
    target_rank: usize,
    nbiter: usize,
    nbpass: usize,
    skip_frac: f64,
    symetric: bool,
    centric: bool,
) -> PyResult<Vec<f64>> {
    let (trimat, _) = load_csv(csv, symetric).map_err(to_py_err)?;
    let csr = trimat.to_csr();
    let params = HopeParams::new(
        HopeMode::ADA,
        RangeApproxMode::RANK(RangeRank::new(target_rank, nbiter)),
        1.0,
    );
    let f = move |t: TriMatI<f64, usize>| {
        let mut h = Hope::new(params, t);
        h.embed().unwrap()
    };
    let auc = link::estimate_auc(&csr, nbpass, skip_frac, symetric, &f);
    if centric {
        let c_auc = link::estimate_centric_auc(&csr, nbpass, skip_frac, symetric, &f);
        info!("centric AUC = {:?}", c_auc);
    }
    Ok(auc)
}

#[pyfunction]
#[pyo3(signature = (csv, epsil, maxrank, blockiter, nbpass=1, skip_frac=0.2, symetric=true, centric=false))]
fn validate_hope_precision(
    csv: &str,
    epsil: f64,
    maxrank: usize,
    blockiter: usize,
    nbpass: usize,
    skip_frac: f64,
    symetric: bool,
    centric: bool,
) -> PyResult<Vec<f64>> {
    let (trimat, _) = load_csv(csv, symetric).map_err(to_py_err)?;
    let csr = trimat.to_csr();
    let params = HopeParams::new(
        HopeMode::ADA,
        RangeApproxMode::EPSIL(RangePrecision::new(epsil, blockiter, maxrank)),
        1.0,
    );
    let f = move |t: TriMatI<f64, usize>| {
        let mut h = Hope::new(params, t);
        h.embed().unwrap()
    };
    let auc = link::estimate_auc(&csr, nbpass, skip_frac, symetric, &f);
    if centric {
        let c_auc = link::estimate_centric_auc(&csr, nbpass, skip_frac, symetric, &f);
        info!("centric AUC = {:?}", c_auc);
    }
    Ok(auc)
}

#[pyfunction]
#[pyo3(signature = (csv, decay, dim, nbiter, nbpass=1, skip_frac=0.2, symetric=true, centric=false))]
fn validate_sketching(
    csv: &str,
    decay: f64,
    dim: usize,
    nbiter: usize,
    nbpass: usize,
    skip_frac: f64,
    symetric: bool,
    centric: bool,
) -> PyResult<Vec<f64>> {
    let (trimat, _) = load_csv(csv, symetric).map_err(to_py_err)?;
    let csr = trimat.to_csr();
    let params = NodeSketchParams {
        sketch_size: dim,
        decay,
        nb_iter: nbiter,
        symetric,
        parallel: true,
    };

    let auc = if params.symetric {
        let f = move |t: TriMatI<f64, usize>| {
            let mut ns = NodeSketch::new(params, t);
            ns.embed().unwrap()
        };
        link::estimate_auc(&csr, nbpass, skip_frac, symetric, &f)
    } else {
        let f = move |t: TriMatI<f64, usize>| {
            let mut ns = NodeSketchAsym::new(params, t);
            ns.embed().unwrap()
        };
        link::estimate_auc(&csr, nbpass, skip_frac, symetric, &f)
    };

    if centric {
        info!("(centric validation not implemented for sketching)");
    }
    Ok(auc)
}

/* ----------------------------  VCMPR  ----------------------------------- */

#[pyfunction]
#[pyo3(signature = (csv, target_rank, nbiter, nbpass=1, nb_edges=10, skip_frac=0.2, symetric=true))]
fn estimate_vcmpr_hope_rank(
    csv: &str,
    target_rank: usize,
    nbiter: usize,
    nbpass: usize,
    nb_edges: usize,
    skip_frac: f64,
    symetric: bool,
) -> PyResult<()> {
    let (trimat, _) = load_csv(csv, symetric).map_err(to_py_err)?;
    let csr = trimat.to_csr();
    let params = HopeParams::new(
        HopeMode::ADA,
        RangeApproxMode::RANK(RangeRank::new(target_rank, nbiter)),
        1.0,
    );
    let f = move |t: TriMatI<f64, usize>| {
        let mut h = Hope::new(params, t);
        h.embed().unwrap()
    };
    link::estimate_vcmpr(&csr, nbpass, nb_edges, skip_frac, symetric, &f);
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (csv, decay, dim, nbiter, nbpass=1, nb_edges=10, skip_frac=0.1, symetric=true))]
fn estimate_vcmpr_sketching(
    csv: &str,
    decay: f64,
    dim: usize,
    nbiter: usize,
    nbpass: usize,
    nb_edges: usize,
    skip_frac: f64,
    symetric: bool,
) -> PyResult<()> {
    let (trimat, _) = load_csv(csv, symetric).map_err(to_py_err)?;
    let csr = trimat.to_csr();
    let params = NodeSketchParams {
        sketch_size: dim,
        decay,
        nb_iter: nbiter,
        symetric,
        parallel: true,
    };
    if params.symetric {
        let f = move |t: TriMatI<f64, usize>| {
            let mut ns = NodeSketch::new(params, t);
            ns.embed().unwrap()
        };
        link::estimate_vcmpr(&csr, nbpass, nb_edges, skip_frac, symetric, &f);
    } else {
        let f = move |t: TriMatI<f64, usize>| {
            let mut ns = NodeSketchAsym::new(params, t);
            ns.embed().unwrap()
        };
        link::estimate_vcmpr(&csr, nbpass, nb_edges, skip_frac, symetric, &f);
    }
    Ok(())
}

/* ----------------------------------------------------------------------- */
/* MODULE DEFINITION                                                       */
/* ----------------------------------------------------------------------- */

#[pymodule]
fn graphembed_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = env_logger::try_init(); // ignore “already initialised”

    /* Embedding */
    m.add_function(wrap_pyfunction!(embed_hope_rank, m)?)?;
    m.add_function(wrap_pyfunction!(embed_hope_precision, m)?)?;
    m.add_function(wrap_pyfunction!(embed_sketching, m)?)?;

    /* Validation */
    m.add_function(wrap_pyfunction!(validate_hope_rank, m)?)?;
    m.add_function(wrap_pyfunction!(validate_hope_precision, m)?)?;
    m.add_function(wrap_pyfunction!(validate_sketching, m)?)?;

    /* VCMPR */
    m.add_function(wrap_pyfunction!(estimate_vcmpr_hope_rank, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_vcmpr_sketching, m)?)?;

    Ok(())
}
