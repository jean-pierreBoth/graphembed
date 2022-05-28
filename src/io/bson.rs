//! file to do bson io for embedding results

/// 
///  Data are formatted in a bson Document, each value as a key.
/// 
///  The encoding is done in 3 parts:
/// 1. A header structure with key "header". The structure is described below [Header](BsonHeader)
/// - a version index
/// - base type essentially usize , f32 or f64 as a String. key is type_name
/// - symetric or asymetric flag
/// - dimension of vectors
/// - number of vectors
/// 
/// 2. The embedded arrays one or two depending on asymetry
/// - loop on number of vectors
///     each vector has a key corresponding to its index and a tag corresponding to OUT (0) or IN
///     so the first vector of embedding has key "0,0" , the second "1,0"
/// - if embedding is asymetric the first loop gives outgoing (or source) representation of node
///   and there is another loop giving ingoing or target) representation of node with key made by IN encoding
///     so first vector of the second group of embedded vectors corresponding to nodes as target has key
///     "0,1" the second vector has key "1,1"
/// 
/// 3. The nodeindexation is encoded in a subdocument associated to key indexation
/// 
///    each nodeid is encoded as string  providing a key in document indeaxtion associated to the node rank as i64.
/// 
/// 
/// 

use anyhow::{anyhow};

use std::fs::{OpenOptions};
use std::path::{Path};
use std::io::{BufWriter, BufReader};

// to convert NodeId to a string, so NodeId must satisfy NodeId : ToString 
// this  requires NodeId :  Display + ?Sized to get auto implementation
use std::string::ToString;

// for serilaizatio, desreialization
use bson::{bson, DeserializerOptions,  Bson, Document};
use serde::{Serialize, Deserialize};


use num::cast::FromPrimitive;
use ndarray::{Array1,Array2, ArrayView1};

use crate::embedding::*;
use crate::tools::edge::{IN,OUT};


/// This structure defines the header of the bson document
#[derive(Debug,Serialize, Deserialize)]
pub struct BsonHeader {
    /// version of dump format
    pub version : i64,
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
        BsonHeader{version : 1, symetric, type_name, dimension, nbdata}
    }
} // end of impl BsonHeader


///
pub fn bson_dump<F, NodeId, EmbeddedData>(embedding : &Embedding<F, NodeId, EmbeddedData>, fname : &String) -> Result<(), anyhow::Error>
    where   NodeId : std::hash::Hash + std::cmp::Eq + std::fmt::Display,  
            EmbeddedData : EmbeddedT<F> ,
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
        "version": 1 as i64,
        "symetric":embedded.is_symetric(),
        "type_name": std::any::type_name::<F>(),  // TODO must be simplified
        "dimension": dim,
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
        key.push_str(",");
        key.push_str(&OUT.to_string());
        // TODO we should store a nodeId to NodeId should satisfy NodeId : ToString ? or separate , or just reload csv file
        doc.insert(key, data);
    }
    // if asymetric we must dump in part
    if !embedded.is_symetric() {
        log::debug!("asymetric part to bson... ");
        for i in 0..nbdata as usize {
            let data : Vec<Bson> = embedded.get_embedded_node(i, IN).iter().map(|x| bson::to_bson(x).unwrap()).collect();
            let ival : i64 = FromPrimitive::from_usize(i).unwrap();
            let mut key = ival.to_string();
            key.push_str(",");
            key.push_str(&IN.to_string());
            doc.insert(key, data);
        }
        log::debug!("asymetric part converted to bson");
    }
    log::debug!("dumping NodeIndexation");
    // We dump nodeindexation as a document with 
    // each key being nodeid converted to a String
    let mut bson_indexation = Document::new();
    let node_indexation = embedding.get_node_indexation();
    for i in 0..node_indexation.len() {
        let node_id = node_indexation.get_index(i).unwrap();
        bson_indexation.insert(node_id.to_string(), i as i64);
    }
    doc.insert("indexation", bson_indexation);
    log::debug!("NodeIndexation bson encoded");
    // write document
    let res = doc.to_writer(bufwriter);

    if res.is_err() {
        log::info!("dump bson in {} done", path.display());
        return Err(anyhow!("dump of bson failed: {}", res.err().unwrap()));
    }

 
    //
    Err(anyhow!("dump of bson not yet"))
}  // end of bson_dump


fn get_vector<F>(bsondata : &Bson) -> Result<Array1<F>, anyhow::Error>  {

    Err(anyhow!("not yet"))
} // end of get_vector



/// returns the bson header of an embedding
/// This can be useful to retrieve the type of the embedding
pub fn get_bson_header(fname : &String) -> Result<BsonHeader, anyhow::Error> {
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
    let header: BsonHeader = bson::from_bson(bson_header).unwrap();
    return Ok(header);
} // end of get_bson_header



pub fn bson_load<'a, F, NodeId, EmbeddedData>(fname : &String) -> Result<Embedding<F, NodeId, EmbeddedData>, anyhow::Error>
    where   NodeId : std::hash::Hash + std::cmp::Eq,  EmbeddedData : EmbeddedT<F> ,
            F : num_traits::Zero + Clone + serde::de::DeserializeOwned,
            NodeId : ToString {
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
    let header: BsonHeader = bson::from_bson(bson_header).unwrap();
    log::debug!("header : {:?}", header);
    if header.version != 1 {
        log::error!("header format version : {}", header.version);
        return Err(anyhow!("format version error, inconsistent with header"));
    }
    // now reload data. First part is default OUT part
    let nb_data : usize =  FromPrimitive::from_i64(header.nbdata).unwrap();
    let dim : usize = FromPrimitive::from_i64(header.dimension).unwrap();
    let type_name = std::any::type_name::<F>();
    if header.type_name != std::any::type_name::<F>() {
        log::error!("header as type name : {}, reloading with : {}", header.type_name, type_name);
        return Err(anyhow!("type error, inconsistent with header"));
    }
    let mut out_array = Array2::<F>::zeros((0, dim));
    for i in 0..nb_data {
        let mut key = i.to_string();
        key.push_str(",");
        key.push_str(&OUT.to_string());
        let res =  doc.get(&key);
        if res.is_none() {
            log::error!("could get record for key {:?}", key);
            return Err(anyhow!("could get record for key {:?}", key));
        }
        // check key
        let options = DeserializerOptions::builder().human_readable(false).build();
        let data_1d : Vec<F> = bson::from_bson_with_options(res.unwrap().clone(), options).unwrap();
        let res = out_array.push_row(ArrayView1::from(data_1d.as_slice()));
        if res.is_err() {
            return Err(anyhow!("could not insert OUT array vector {:?}", i));
        }
    }
    log::debug!("got OUT part of embedding");
    // 
    if !header.symetric {
        let mut in_array = Array2::<F>::zeros((0, dim));
        log::debug!("asymetric embedding, decoding IN part of embedding");
        for i in 0..nb_data {
            let mut key = i.to_string();
            key.push_str(",");
            key.push_str(&IN.to_string());
            let res =  doc.get(&key);
            if res.is_none() {
                log::error!("could get record for key {:?}", key);
                return Err(anyhow!("could get record for key {:?}", key));
            }        
            let options = DeserializerOptions::builder().human_readable(false).build();
            let data_1d : Vec<F> = bson::from_bson_with_options(res.unwrap().clone(), options).unwrap();
            let res = in_array.push_row(ArrayView1::from(data_1d.as_slice()));
            if res.is_err() {
                return Err(anyhow!("could not insert IN array vector {:?}", i));
            }
        } 
    }  
    log::debug!("finished bsoon decoding of embedding");
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