//! file to do bson io for embedding results

/// 
///  Data are formatted in a bson Document, each value as a key.
/// 
/// - base type essentially usize , f32 or f64 as a String see std::any::type_name
/// - symetric or asymetric flag
/// - dimension of vectors
/// - number of vectors
/// - loop on number of vectors
///     each vector has a key corresponding to its index
/// 
/// 
/// 
/// 

use log::*;
use anyhow::{anyhow};

use serde::{Serialize, Deserialize};
use std::fs::{OpenOptions};
use std::path::{Path};
use std::str::FromStr;
use std::io::Cursor;

use std::io::{BufWriter};
use bson::{bson, Bson, Document};

use std::any::type_name;

use num::cast::FromPrimitive;

use crate::embedding::*;

#[derive(Serialize, Deserialize)]
pub(crate) struct BsonHeader {
    symetric : bool,
    type_name : String,
    dimension : usize,
    nbdata : usize,
} // end of BsonHeader


impl BsonHeader {
    pub fn new(symetric : bool , type_name : String, dimension : usize, nbdata : usize) -> Self {
        BsonHeader{symetric, type_name, dimension, nbdata}
    }
} // end of impl BsonHeader


///
pub fn bson_dump<F>(embedded : &Embedded<F>, fname : &String) -> Result<(), anyhow::Error> {
    //
    log::debug!("entering bson_dump");
    //
    let path = Path::new(fname);
    let fileres = OpenOptions::new().write(true).open(&path);
    let file;
    if fileres.is_ok() {
        file = fileres.unwrap();
    }
    else {
        return Err(anyhow!("could not open file : {}", fname));
    }
    let mut bufwriter = BufWriter::new(file);
    let mut doc = Document::new();

    // dump header part
    let dim : i64 = FromPrimitive::from_usize(embedded.get_dimension()).unwrap();
    let nbdata : i64 =  FromPrimitive::from_usize(embedded.get_nb_nodes()).unwrap();

    let header = bson!({
        "symetry":true,
        "type": std::any::type_name::<F>(),  // TODO must be simplified
        "dim": dim,
        "nbdata": nbdata
    }
    );
    doc.insert("header", header);
    // now loop on data vectors

    //
    return Err(anyhow!("not yet"));
}  // end of bson_dump


