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
/// - if embedding is asymetric the first loop gives outgoing (or source) representation of node
///   and there is another loop giving ingoing or target) representation of node
/// 
///  The document begins with a structure [Header](BsonHeader) associated to the key "header"
/// 
/// 

use anyhow::{anyhow};

use serde::{Serialize, Deserialize};
use std::fs::{OpenOptions};
use std::path::{Path};
//use std::io::Cursor;

use std::io::{BufWriter, BufReader};
use bson::{bson, Bson, Document};


use num::cast::FromPrimitive;

use crate::embedding::*;
use crate::tools::edge::{IN,OUT};


/// This structure defines the header of the bson document
#[derive(Serialize, Deserialize)]
pub struct BsonHeader {
    /// true if embedding is symetric
    pub symetric : bool,
    /// encodes type of vectors used in the embedding. Must be f32 f64 or usze
    pub type_name : String,
    /// dimension of the embedding (length of vectors)
    pub dimension : i64,
    /// number of vectors.
    pub nbdata : i64,
} // end of BsonHeader


impl BsonHeader {
    pub fn new(symetric : bool , type_name : String, dimension : i64, nbdata : i64) -> Self {
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
    let fileres = OpenOptions::new().write(true).create(true).open(path);
    let file;
    if fileres.is_ok() {
        file = fileres.unwrap();
    }
    else {
        return Err(anyhow!("could not open file : {}", path.display()));
    }
    let bufwriter = BufWriter::new(file);
    let mut doc = Document::new();

    let embedded = embedding.get_embedded_data();
    // dump header part
    let dim : i64 = FromPrimitive::from_usize(embedded.get_dimension()).unwrap();
    let nbdata : i64 =  FromPrimitive::from_usize(embedded.get_nb_nodes()).unwrap();
    // we could allocate a BsonHeader and call bson::to_bson(&bson_header).unwrap() but for C decoder ...
    let bson_header = bson!({
        "symetry":embedded.is_symetric(),
        "type": std::any::type_name::<F>(),  // TODO must be simplified
        "dim": dim,
        "nbdata": nbdata
        }
    ); 
    doc.insert("header", bson_header);
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
        return Ok(());
    }
    else {
        return Err(anyhow!("dump of bson failed: {}", res.err().unwrap()));
    }
    //
}  // end of bson_dump



pub fn bson_load<F, NodeId, EmbeddedData>(fname : &String) -> Result<Embedding<F, NodeId, EmbeddedData>, anyhow::Error>
    where   NodeId : std::hash::Hash + std::cmp::Eq,  EmbeddedData : EmbeddedT<F> ,
            F : Serialize {
    //
    log::debug!("entering bson_dump");
    //
    let path = Path::new(fname);
    let fileres = OpenOptions::new().read(true).open(&path);
    let file;
    if fileres.is_ok() {
        file = fileres.unwrap();
    }
    else {
        log::error!("reload of bson dump failed");
        return Err(anyhow!("reloadfailed: {}", fileres.err().unwrap()));
    }
    let mut bufreader = BufReader::new(file);
    let res = Document::from_reader(&mut bufreader);
    if res.is_err() {
        log::error!("could load document from file {}", path.display());
        return Err(anyhow!(res.err().unwrap()));
    }
    let doc = res.unwrap();
    let res =  doc.get("header");
    if res.is_none() {
        log::error!("could load header from file {}", path.display());
        return Err(anyhow!("could not find header in document"));
    }
    let bson_header = res.unwrap().clone();  // TODO avoid clone ?
    // now decode our fields
    let _header: BsonHeader = bson::from_bson(bson_header).unwrap();
    //
    return Err(anyhow!("not yet"));
} // end of bson_load



#[cfg(test)]
mod tests {

    use super::*;
    use crate::prelude::*;

    fn log_init_test() {
        let res = env_logger::builder().is_test(true).try_init();
        if res.is_err() {
            println!("could not init log");
        }
    }  // end of log_init_test





#[test]
    fn test_bson_moreno() {
        log_init_test();
        //
        let path = Path::new(crate::DATADIR).join("moreno_lesmis").join("out.moreno_lesmis_lesmis");
        log::debug!("\n\n test_weighted_csv_to_trimat, loading file {:?}", path);
        let header_size = crate::io::csv::get_header_size(&path);
        assert_eq!(header_size.unwrap(),2);
        println!("\n\n test_weighted_csv_to_trimat, data : {:?}", path);
        //
        let trimat_res  = csv_to_trimat::<f64>(&path, false, b' ');
        if let Err(err) = &trimat_res {
            eprintln!("ERROR: {}", err);
            assert_eq!(1,0);
        }
        let (trimat, node_indexation)  = trimat_res.unwrap();
        // embed
        let sketch_size = 15;
        let decay = 0.1;
        let nb_iter = 2;
        let parallel = false;
        let symetric = true;
        let sketching_params = NodeSketchParams{sketch_size, decay, nb_iter, symetric, parallel};
        let mut nodesketch = NodeSketch::new(sketching_params, trimat);
        let embedding = Embedding::new(node_indexation, &mut nodesketch);
        if embedding.is_err() {
            log::error!("nodesketch embedding failed error : {:?}", embedding.as_ref().err());
            std::process::exit(1);
        };
        let embedding = embedding.unwrap();
        // now we can do a bson dump
        let dumpfname = String::from("moreno_bson");
        let bson_res = bson_dump(&embedding, &String::from("moreno_bson"));
        if bson_res.is_err() {
            log::error!("bson dump in file {} failed", &dumpfname);
            log::error!("error returned : {:?}", bson_res.err().unwrap());
            assert_eq!(1,0);
        } 
        //
        log::info!("trying reload from {}", &dumpfname);
        let reloaded = bson_load::<usize, usize, Embedded<usize>>(&dumpfname);
        if reloaded.is_err() {
            log::error!("reloading of bson from {} failed", dumpfname);
        }
        // now we must compare

    } // end of test_bson_mini

} // end of mod tests