//! module to do bson io for embedding results
//!
//!  Data are formatted in a bson Document, each value has a key.
//!
//!  The encoding is done in 3 parts:
//! 1. A header structure with key "header". The structure is described below see struct [Header](EmbeddedBsonHeader)
//! - a version index
//! - base type name essentially f32,f64 or usize encoded as a String. key is type_name.  
//!   **Beware that as bson requires usize to be encoded as i64, an independant implementation of a reload
//!   will need to parse as i64 in the Nodesketch embedding which uses usize vectors. (See sources)**
//! - symetric or asymetric flag
//! - dimension of vectors
//! - number of vectors
//!
//! 2. The embedded arrays, one or two depending on asymetry
//!     - loop on number of vectors
//!       each vector has a key corresponding to its index and a tag corresponding to OUT (0) or IN (1)
//!       so the first vector of embedding has key "0,0" , the second "1,0"
//!     - if embedding is asymetric the first loop gives outgoing (or source) representation of node
//!       and there is another loop giving ingoing or target) representation of node with key made by IN encoding
//!       so first vector of the second group of embedded vectors corresponding to nodes as target has key
//!       "0,1" the second vector has key "1,1"
//!
//! 3. The nodeindexation can also be encoded in a subdocument associated to key *"indexation"*.
//!    The dump of nodeindexation is not mandatory as it can be retrieved by loading the original graph again.  
//!    The presence of nodeindexation can be tested by checking if the main document has ky *"indexation"*.  
//!    If the sub-document is present : each nodeid is encoded as string  providing a key in document indexation associated to the node rank as i64.
//!  

// Note : a Bson document must not be larger than 16Mb!
// So we need to have many Documents in the file dumped

use anyhow::anyhow;

use std::fs::OpenOptions;
use std::io::{BufReader, BufWriter};
use std::path::Path;

// to convert NodeId to a string, so NodeId must satisfy NodeId : ToString
// this  requires NodeId :  Display + ?Sized to get auto implementation
use std::str::FromStr;

// for serilaizatio, desreialization
use bson::{Bson, Document, bson};
use serde::{Deserialize, Serialize};

use indexmap::IndexSet;
use ndarray::{Array2, ArrayView1};
use num::cast::FromPrimitive;

use crate::embed::tools::edge::{IN, OUT};
use crate::embedding::*;
use crate::io;

/// This structure defines the header of the bson document
#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddedBsonHeader {
    /// version of dump format
    pub version: i64,
    /// true if embedding is symetric
    pub symetric: bool,
    /// encodes type of vectors used in the embedding. Must be f32 f64 or usze
    pub type_name: String,
    /// dimension of the embedding (length of vectors)
    pub dimension: i64,
    /// number of vectors.
    pub nbdata: i64,
} // end of EmbeddedBsonHeader

impl EmbeddedBsonHeader {
    pub fn new(symetric: bool, type_name: String, dimension: i64, nbdata: i64) -> Self {
        EmbeddedBsonHeader {
            version: 1,
            symetric,
            type_name,
            dimension,
            nbdata,
        }
    }
} // end of impl EmbeddedBsonHeader

/// dump an embedding in bson format in filename fname.  
/// If dump_indexation is true, nodeindexation will also be dumped, and retrieved from bson file.
/// The dump consists in a header document. Then each node is dumped in its document (a bson document must less than 16Mb)
/// The last document contains the indexation if dumped is asked for.
///
pub fn bson_dump<F, NodeId, EmbeddedData>(
    embedding: &Embedding<F, NodeId, EmbeddedData>,
    output: &io::output::Output,
) -> Result<(), anyhow::Error>
where
    NodeId: std::hash::Hash + std::cmp::Eq + std::fmt::Display,
    EmbeddedData: EmbeddedT<F>,
    F: Serialize,
{
    //
    log::info!("entering bson_dump");
    //
    let path = Path::new(output.get_output_name());
    let fileres = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path);
    let file = if fileres.is_ok() {
        fileres.unwrap()
    } else {
        return Err(anyhow!("could not open file : {}", path.display()));
    };
    let mut bufwriter = BufWriter::new(file);
    let mut doc = Document::new();

    let embedded = embedding.get_embedded_data();
    log::info!("\t symetry embedding : {}", embedded.is_symetric());
    // dump header part
    let dim: i64 = FromPrimitive::from_usize(embedded.get_dimension()).unwrap();
    let nbdata: i64 = FromPrimitive::from_usize(embedded.get_nb_nodes()).unwrap();
    // we could allocate a EmbeddedBsonHeader and call bson::to_bson(&bson_header).unwrap() but for C decoder ...
    let bson_header = bson!({
        "version": 1_i64,
        "symetric":embedded.is_symetric(),
        "type_name": std::any::type_name::<F>(),  // TODO must be simplified
        "dimension": dim,
        "nbdata": nbdata
        }
    );
    doc.insert("header", bson_header);
    let res = doc.to_writer(&mut bufwriter);
    if res.is_err() {
        log::error!("dump header  bson in {} done", path.display());
        return Err(anyhow!("dump of bson failed: {}", res.err().unwrap()));
    }
    // now loop on data vectors
    for i in 0..nbdata as usize {
        let mut doc = Document::new();
        let data: Vec<Bson> = embedded
            .get_embedded_node(i, OUT)
            .iter()
            .map(|x| bson::to_bson(x).unwrap())
            .collect();
        let ival: i64 = FromPrimitive::from_usize(i).unwrap();
        let mut key = ival.to_string();
        key.push(',');
        key.push_str(&OUT.to_string());
        doc.insert(key, data);
        let res = doc.to_writer(&mut bufwriter);
        if res.is_err() {
            log::error!("bson dump error OUT , in node {i}");
            return Err(anyhow!(
                "bson dump error for OUT node {i} {}",
                res.err().unwrap()
            ));
        }
    }
    // if asymetric we must dump in part
    if !embedded.is_symetric() {
        log::debug!("asymetric part to bson... ");
        for i in 0..nbdata as usize {
            let mut doc = Document::new();
            let data: Vec<Bson> = embedded
                .get_embedded_node(i, IN)
                .iter()
                .map(|x| bson::to_bson(x).unwrap())
                .collect();
            let ival: i64 = FromPrimitive::from_usize(i).unwrap();
            let mut key = ival.to_string();
            key.push(',');
            key.push_str(&IN.to_string());
            doc.insert(key, data);
            let res = doc.to_writer(&mut bufwriter);
            if res.is_err() {
                log::error!("bson dump error  in node IN  {i}");
                return Err(anyhow!(
                    "bson dump error for IN node {i} {}",
                    res.err().unwrap()
                ));
            }
        }
        log::debug!("asymetric part converted to bson");
    }
    // We dump nodeindexation as a document with
    // each key being nodeid converted to a String
    if output.get_indexation() {
        log::info!("\t dumping NodeIndexation");
        let mut bson_indexation = Document::new();
        let node_indexation = embedding.get_node_indexation();
        for i in 0..node_indexation.len() {
            let node_id = node_indexation.get_index(i).unwrap();
            bson_indexation.insert(node_id.to_string(), i as i64);
        }
        // write indexation
        let res = bson_indexation.to_writer(&mut bufwriter);
        if res.is_err() {
            log::error!("dump bson in {} done", path.display());
            return Err(anyhow!("dump of bson failed: {}", res.err().unwrap()));
        }
        log::debug!("\t NodeIndexation bson encoded");
    } // end dump indexation
    //
    log::info!("bson dump in file {} finished", path.display());
    //
    Ok(())
    //    Err(anyhow!("dump of bson not yet"))
} // end of bson_dump

/// returns the bson header of an embedding.  
/// This can be useful to retrieve the type of the embedding (dumped via a call to std::any::type_name::\<F\>()).
/// The type will be "f64", "f32" or "i64" as Bson imposes the conversion of usize to i64
pub fn get_bson_header(fname: &String) -> Result<EmbeddedBsonHeader, anyhow::Error> {
    let path = Path::new(fname);
    log::info!("get_bson_header: trying to open file : {:?}", path);
    let fileres = OpenOptions::new().read(true).open(path);
    let file;
    if fileres.is_ok() {
        file = fileres.unwrap();
    } else {
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
    let res = doc.get("header");
    if res.is_none() {
        log::error!("could load header from file {}", path.display());
        return Err(anyhow!("could not find header in document"));
    }
    let bson_header = res.unwrap().clone(); // TODO avoid clone ?
    let header: EmbeddedBsonHeader = bson::from_bson(bson_header).unwrap();
    log::info!(" bson header reloaded");
    Ok(header)
} // end of get_bson_header

/// The structure returned by bson_load.
pub struct EmbeddedBsonReload<F, NodeId> {
    /// The minimal embedded data when graph is symetric
    pub(crate) out_embedded: Array2<F>,
    /// If grap is asymetric we get embedding of nodes as targets (or destinations) in_embedded
    pub(crate) in_embedded: Option<Array2<F>>,
    /// If nodeindexation was dumped in bson
    pub(crate) node_indexation: Option<IndexSet<NodeId>>,
} // end of EmbeddedBsonReload

impl<F, NodeId> EmbeddedBsonReload<F, NodeId> {
    pub fn new(
        out_embedded: Array2<F>,
        in_embedded: Option<Array2<F>>,
        node_indexation: Option<IndexSet<NodeId>>,
    ) -> Self {
        EmbeddedBsonReload {
            out_embedded,
            in_embedded,
            node_indexation,
        }
    }
    /// returns embedded data. If asymetric embedding it is embedded as a source.
    /// If embedding is symetric it is the whole embedding.
    pub fn get_out_embedded(&self) -> &Array2<F> {
        &self.out_embedded
    }
    /// returns node indexation if present
    pub fn get_node_indexation(&self) -> Option<&IndexSet<NodeId>> {
        self.node_indexation.as_ref()
    }
    /// If the embedding dumped is asymetric this returns an array giving the embedding as a target (or destination) node.
    pub fn get_in_embedded(&self) -> Option<&Array2<F>> {
        self.in_embedded.as_ref()
    }
} // enf of impl EmbeddedBsonReload

/// reloads embedded data from a previous bson dump and returns a EmbeddedBsonReload structure.  
/// The structure  Embedded or EmbeddedAsym can be reconstituted from it (or with a graph reload to get nodeindexation)
///
pub fn bson_load<F, NodeId, EmbeddedData>(
    fname: &str,
) -> Result<EmbeddedBsonReload<F, NodeId>, anyhow::Error>
where
    NodeId: std::hash::Hash + std::cmp::Eq,
    EmbeddedData: EmbeddedT<F>,
    F: num_traits::Zero + Clone + serde::de::DeserializeOwned,
    NodeId: ToString + FromStr,
{
    //
    log::info!("entering bson_load, file name : {:?}", fname);
    //
    let path = Path::new(fname);
    let fileres = OpenOptions::new().read(true).open(path);
    let file;
    if fileres.is_ok() {
        file = fileres.unwrap();
        log::debug!("bson_load, path name : {:?}, file ok , unwrap done", path);
    } else {
        log::error!("reload of bson dump failed");
        return Err(anyhow!("reloadfailed: {}", fileres.err().unwrap()));
    }
    let mut bufreader = BufReader::new(file);
    // load header
    let res = Document::from_reader(&mut bufreader);
    if res.is_err() {
        log::error!("could not load document from file {}", path.display());
        return Err(anyhow!(res.err().unwrap()));
    }
    let doc = res.unwrap();
    let res = doc.get("header");
    if res.is_none() {
        log::error!("could load header from file {}", path.display());
        return Err(anyhow!("could not find header in document"));
    }
    log::debug!("\t found header key");
    let bson_header = res.unwrap().clone(); // TODO avoid clone ?
    // now decode our fields
    let header: EmbeddedBsonHeader = bson::from_bson(bson_header).unwrap();
    log::info!("header : {:?}", header);
    if header.version != 1 {
        log::error!("header format version : {}", header.version);
        return Err(anyhow!("format version error, inconsistent with header"));
    }
    // now reload data. First part is default OUT part
    let nb_data: usize = FromPrimitive::from_i64(header.nbdata).unwrap();
    log::debug!("bson_load , nb_data = {nb_data}");
    let dim: usize = FromPrimitive::from_i64(header.dimension).unwrap();
    log::debug!("bson_load , dim : {dim}");
    let type_name = std::any::type_name::<F>();
    if header.type_name != std::any::type_name::<F>() {
        log::error!(
            "header as type name : {}, reloading with : {}",
            header.type_name,
            type_name
        );
        return Err(anyhow!("type error, inconsistent with header"));
    }
    let mut out_array = Array2::<F>::zeros((0, dim));
    for i in 0..nb_data {
        // we have one document for each node
        let res = Document::from_reader(&mut bufreader);
        if res.is_err() {
            log::error!(
                "could not load document  for node {i} from file {}",
                path.display()
            );
            return Err(anyhow!(res.err().unwrap()));
        }
        let doc = res.unwrap();

        let mut key = i.to_string();
        key.push(',');
        key.push_str(&OUT.to_string());
        let res = doc.get(&key);
        if res.is_none() {
            log::error!("could not get record for key {:?}", key);
            return Err(anyhow!("could not get record for key {:?}", key));
        }
        // check key
        //        let options = DeserializerOptions::builder().human_readable(false).build();
        let data_1d_res = bson::from_bson(res.unwrap().clone());
        if data_1d_res.is_err() {
            log::info!(
                "\t bson decoding error for node {i}, key : {key}, err : {:?}",
                data_1d_res.err()
            );
            panic!("bson decoding error for OUT node {i}");
        }
        let data_1d: Vec<F> = data_1d_res.unwrap();
        let res = out_array.push_row(ArrayView1::from(data_1d.as_slice()));
        if res.is_err() {
            return Err(anyhow!("could not insert OUT array vector {:?}", i));
        }
        log::debug!("\t decoded OUT array vector {:?}", i);
    }
    log::info!("\t got OUT part of embedding");
    //
    let mut in_array_opt: Option<Array2<F>> = None;
    if !header.symetric {
        let mut in_array = Array2::<F>::zeros((0, dim));
        log::info!("\t asymetric embedding, decoding IN part of embedding");
        for i in 0..nb_data {
            // we have one document for each node
            let res = Document::from_reader(&mut bufreader);
            if res.is_err() {
                log::error!(
                    "could not load document  for node IN {i} from file {}",
                    path.display()
                );
                return Err(anyhow!(res.err().unwrap()));
            }
            let doc = res.unwrap();
            let mut key = i.to_string();
            key.push(',');
            key.push_str(&IN.to_string());
            let res = doc.get(&key);
            if res.is_none() {
                log::error!("could get record for key {:?}", key);
                return Err(anyhow!("could get record for key {:?}", key));
            }
            //            let options = DeserializerOptions::builder().human_readable(false).build();
            let data_1d: Vec<F> = bson::from_bson(res.unwrap().clone()).unwrap();
            let res = in_array.push_row(ArrayView1::from(data_1d.as_slice()));
            if res.is_err() {
                return Err(anyhow!("could not insert IN array vector {:?}", i));
            }
            log::debug!("decoded IN array vector {:?}", i);
        }
        in_array_opt = Some(in_array);
    }
    log::info!("\t finished bson decoding of embedded vectors");
    log::info!("\t bson load checking for key indexation");
    // trying node indexation
    let res = Document::from_reader(&mut bufreader);
    if res.is_err() {
        log::info!(
            "could not find indexation document from file {}, err : {:?}",
            path.display(),
            res.err()
        );
        Ok(EmbeddedBsonReload::new(out_array, in_array_opt, None))
    } else {
        log::info!("\t found document , node indexation");
        let bson_indexation = res.unwrap();
        let mut node_indexation = IndexSet::<NodeId>::with_capacity(bson_indexation.len());
        let mut index_iter = bson_indexation.iter();
        while let Some(item) = index_iter.next() {
            let res = NodeId::from_str(item.0);
            let node_id = match res {
                Ok(node_id) => node_id,
                Err(_e) => {
                    log::error!("could get node rank for node_id {}", item.0);
                    return Err(anyhow!("could get node rank for node_id {}", item.0));
                }
            };
            let res = item.1.as_i64();
            if res.is_none() {
                log::error!("could get node rank for node_id {}", item.0);
                return Err(anyhow!("could get node rank for node_id {}", item.0));
            }
            node_indexation.insert(node_id);
        }
        assert_eq!(node_indexation.len(), nb_data);
        Ok(EmbeddedBsonReload::new(
            out_array,
            in_array_opt,
            Some(node_indexation),
        ))
    } // end case we have indexation in bson file
} // end of bson_load

// This function checks equality of embedded and reloaded
#[allow(unused)]
fn check_equality<F, NodeId, EmbeddedData>(
    embedding: &Embedding<F, NodeId, EmbeddedData>,
    reloaded: &EmbeddedBsonReload<F, NodeId>,
) -> Result<bool, anyhow::Error>
where
    NodeId: std::hash::Hash + std::cmp::Eq,
    EmbeddedData: EmbeddedT<F>,
    F: num_traits::Zero + Clone + serde::de::DeserializeOwned + PartialEq + std::fmt::Display,
    NodeId: ToString + FromStr,
{
    let embedded_data = embedding.get_embedded_data();
    let out_reloaded = reloaded.get_out_embedded();
    assert_eq!(
        out_reloaded.dim(),
        (embedded_data.get_nb_nodes(), embedded_data.get_dimension())
    );
    // first chech out as it is the default and is always present
    log::info!("test_bson_moreno checking equality of reload, OUT embedding");
    for i in 0..embedded_data.get_nb_nodes() {
        let vec_e = embedded_data.get_embedded_node(i, OUT);
        for j in 0..embedded_data.get_dimension() {
            if vec_e[j] != out_reloaded[[i, j]] {
                log::error!(
                    " reloaded differ from embedded at vector rank : {}, dim j : {}, embedded : {}, reloaded : {}",
                    i,
                    j,
                    vec_e[j],
                    out_reloaded[[i, j]]
                );
            }
        }
    }
    if !embedded_data.is_symetric() {
        log::info!("test_bson_moreno checking equality of reload : IN embedding");
        // same thing with tag = IN
        let in_reloaded = reloaded.get_in_embedded().unwrap();
        for i in 0..embedded_data.get_nb_nodes() {
            let vec_e = embedded_data.get_embedded_node(i, IN);
            for j in 0..embedded_data.get_dimension() {
                if vec_e[j] != in_reloaded[[i, j]] {
                    log::error!(
                        " reloaded differ from embedded at vector rank : {}, dim j : {}, embedded : {}, reloaded : {}",
                        i,
                        j,
                        vec_e[j],
                        in_reloaded[[i, j]]
                    );
                    return Ok(false);
                }
            }
        }
    } // end check IN in asymetric case
    // check equality of node indexation
    if reloaded.get_node_indexation().is_some() {
        log::debug!("checking node indeation");
        let loaded_indexation = reloaded.get_node_indexation().unwrap();
        let node_indexation = embedding.get_node_indexation();
        assert_eq!(loaded_indexation.len(), node_indexation.len());
        for i in 0..node_indexation.len() {
            let indexed = node_indexation.get_index(i).unwrap();
            let reload_indexed = loaded_indexation.get_index(i).unwrap();
            if indexed != reload_indexed {
                log::error!(
                    "chech equality of node indexation failed at slot i : {} , node_indexation : {}, reloaded {}",
                    i,
                    indexed.to_string(),
                    reload_indexed.to_string()
                );
                return Ok(false);
            }
        }
    } // end case node_indexation
    log::debug!("check_equality exiting");
    Ok(true)
} // end of check equality

#[cfg(test)]
mod tests {

    use super::*;
    use crate::prelude::*;

    fn log_init_test() {
        let res = env_logger::builder().is_test(true).try_init();
        if res.is_err() {
            println!("could not init log");
        }
    } // end of log_init_test

    // test with usize embedding
    #[test]
    fn test_bson_moreno_usize() {
        log_init_test();
        //
        let path = Path::new(crate::DATADIR)
            .join("moreno_lesmis")
            .join("out.moreno_lesmis_lesmis");
        log::debug!("\n\n test_bson_moreno, loading file {:?}", path);
        let header_size = crate::io::csv::get_header_size(&path);
        assert_eq!(header_size.unwrap(), 2);
        println!("\n\n test_weighted_csv_to_trimat, data : {:?}", path);
        //
        let trimat_res = csv_to_trimat::<f64>(&path, false, b' ');
        if let Err(err) = &trimat_res {
            eprintln!("ERROR: {}", err);
            assert_eq!(1, 0);
        }
        let (trimat, node_indexation) = trimat_res.unwrap();
        // embed
        let sketch_size = 15;
        let decay = 0.1;
        let nb_iter = 2;
        let parallel = false;
        let symetric = true;
        let sketching_params = NodeSketchParams {
            sketch_size,
            decay,
            nb_iter,
            symetric,
            parallel,
        };
        let mut nodesketch = NodeSketch::new(sketching_params, trimat);
        let embedding = Embedding::new(node_indexation, &mut nodesketch);
        if embedding.is_err() {
            log::error!(
                "nodesketch embedding failed error : {:?}",
                embedding.as_ref().err()
            );
            std::process::exit(1);
        };
        let embedding = embedding.unwrap();
        // now we can do a bson dump
        let output = io::output::Output::new(
            io::output::Format::BSON,
            true,
            &Some(String::from("moreno_usize.bson")),
        );
        let bson_res = bson_dump(&embedding, &output);
        if bson_res.is_err() {
            log::error!("bson dump in file {} failed", &output.get_output_name());
            log::error!("error returned : {:?}", bson_res.err().unwrap());
            assert_eq!(1, 0);
        }
        //
        log::info!("trying reload from {}", &output.get_output_name());
        let reloaded = bson_load::<usize, usize, Embedded<usize>>(output.get_output_name());
        if reloaded.is_err() {
            log::error!("reloading of bson from {} failed", output.get_output_name());
        }
        let reloaded = reloaded.unwrap();
        //
        let res_equality = check_equality(&embedding, &reloaded);
        match &res_equality {
            Err(_e) => {
                log::error!("check equality encountered error in test_bson_moreno_usize");
                assert_eq!(1, 0);
            }
            Ok(val) => match val {
                false => {
                    log::error!("check equality returned false in test_bson_moreno");
                    assert_eq!(1, 0);
                }
                true => {}
            },
        }
    } // end of test_bson_moreno

    #[test]
    fn test_bson_moreno_f32() {
        log_init_test();
        //
        let path = Path::new(crate::DATADIR)
            .join("moreno_lesmis")
            .join("out.moreno_lesmis_lesmis");
        log::debug!("\n\n test_bson_moreno, loading file {:?}", path);
        let header_size = crate::io::csv::get_header_size(&path);
        assert_eq!(header_size.unwrap(), 2);
        println!("\n\n test_weighted_csv_to_trimat, data : {:?}", path);
        //
        let trimat_res = csv_to_trimat::<f32>(&path, false, b' ');
        if let Err(err) = &trimat_res {
            eprintln!("ERROR: {}", err);
            assert_eq!(1, 0);
        }
        let (trimat, node_indexation) = trimat_res.unwrap();
        // embed
        let hope_m = HopeMode::ADA;
        let decay_f = 0.05;
        //    let range_m = RangeApproxMode::RANK(RangeRank::new(500, 2));
        let range_m = RangeApproxMode::EPSIL(RangePrecision::new(0.1, 10, 300));
        let params = HopeParams::new(hope_m, range_m, decay_f);
        // now we embed
        let mut hope = Hope::new(params, trimat);
        let hope_embedding = Embedding::new(node_indexation, &mut hope).unwrap();
        // now we can do a bson dump
        let output = io::output::Output::new(
            io::output::Format::BSON,
            true,
            &Some(String::from("moreno_f32.bson")),
        );
        let bson_res = bson_dump(&hope_embedding, &output);
        if bson_res.is_err() {
            log::error!("bson dump in file {} failed", &output.get_output_name());
            log::error!("error returned : {:?}", bson_res.err().unwrap());
            assert_eq!(1, 0);
        }
        //
        log::info!("trying reload from {}", &output.get_output_name());
        let reloaded = bson_load::<f32, usize, Embedded<f32>>(output.get_output_name());
        if reloaded.is_err() {
            log::error!("reloading of bson from {} failed", output.get_output_name());
        }
        let reloaded = reloaded.unwrap();
        //
        let res_equality = check_equality(&hope_embedding, &reloaded);
        match &res_equality {
            Err(_e) => {
                log::error!("check equality encountered error in test_bson_moreno");
                assert_eq!(1, 0);
            }
            Ok(val) => match val {
                false => {
                    log::error!("check equality returned false in test_bson_moreno");
                    assert_eq!(1, 0);
                }
                true => {}
            },
        }
    } // end of test_bson_moreno_f32
} // end of mod tests
