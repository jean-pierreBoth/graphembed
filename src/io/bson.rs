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

use anyhow::{anyhow};

use serde::{Serialize, Deserialize};
use std::fs::{OpenOptions};
use std::path::{Path};
//use std::io::Cursor;

use std::io::{BufWriter};
use bson::{bson, Bson, Document};


use num::cast::FromPrimitive;

use crate::embedding::*;
use crate::tools::edge::{IN,OUT};

#[derive(Serialize, Deserialize)]
pub(crate) struct BsonHeader {
    symetric : bool,
    type_name : String,
    dimension : usize,
    nbdata : usize,
} // end of BsonHeader


impl BsonHeader {
#[allow(unused)]
    pub fn new(symetric : bool , type_name : String, dimension : usize, nbdata : usize) -> Self {
        BsonHeader{symetric, type_name, dimension, nbdata}
    }
} // end of impl BsonHeader


///
pub fn bson_dump<F, NodeId, EmbeddedData>(embedding : &Embedding<F, NodeId, EmbeddedData>, fname : &String) -> Result<(), anyhow::Error>
    where   NodeId : std::hash::Hash + std::cmp::Eq,  EmbeddedData : EmbeddedT<F> ,
            F : Serialize {
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
    let bufwriter = BufWriter::new(file);
    let mut doc = Document::new();

    let embedded = embedding.get_embedded_data();
    // dump header part
    let dim : i64 = FromPrimitive::from_usize(embedded.get_dimension()).unwrap();
    let nbdata : i64 =  FromPrimitive::from_usize(embedded.get_nb_nodes()).unwrap();

    let header = bson!({
        "symetry":embedded.is_symetric(),
        "type": std::any::type_name::<F>(),  // TODO must be simplified
        "dim": dim,
        "nbdata": nbdata
    }
    );
    doc.insert("header", header);
    // TODO we should store a nodeId to NodeId should satisfy NodeId : ToString ? or separate , or just reload csv file
    // now loop on data vectors
    for i in 0..nbdata as usize {
        let data : Vec<Bson> = embedded.get_embedded_node(i, OUT).iter().map(|x| bson::to_bson(x).unwrap()).collect();
        let ival : i64 =  FromPrimitive::from_usize(i).unwrap();
        let mut key = ival.to_string();
        key.push_str("-OUT");
        // TODO we should store a nodeId to NodeId should satisfy NodeId : ToString ? or separate , or just reload csv file
        doc.insert(key, data);
    }
    // if asymetric we must dump in part
    if !embedded.is_symetric() {
        for i in 0..nbdata as usize {
            let data : Vec<Bson> = embedded.get_embedded_node(i, IN).iter().map(|x| bson::to_bson(x).unwrap()).collect();
            let ival : i64 =  FromPrimitive::from_usize(i).unwrap();
            let mut key = ival.to_string();
            key.push_str("-IN");
            doc.insert(key, data);
        }
    }
    // write document
    let res = doc.to_writer(bufwriter);
    if res.is_ok() {
        log::info!("dump bson in {} done", path.display());
    }
    //
    return Err(anyhow!("not yet"));
}  // end of bson_dump




#[cfg(test)]
mod tests {

#[allow(unused)]
fn log_init_test() {
    let res = env_logger::builder().is_test(true).try_init();
    if res.is_err() {
        println!("could not init log");
    }
}  // end of log_init_test

}