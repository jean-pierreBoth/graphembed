//! an executable for embedding graph
//! 
//! The main arguments are 
//!  - --csv filename
//!  - --symetric true or false  specifies if the csv file describes a symetric (half of the edges in csv) or asymetric graph.  
//!     If the file is declared symetric, each edge(a,b) is completed upon reading by the edge (b,a).  
//!     Sometimes a symetric graph is fully described in the csv file, then declare the file as asymetric.
//!     It is also possible to declare as symetric an asymetric unweighted graph. It is then symetrised when reading the file.
//!  - --output or -o filename  
//!     This dumps the embedding in a bson file named filename.bson. See module [bson].  
//!     By default an embedding is written in the file **embedding.bson**.
//! 
//!  - a subcommand embedding for a simple embedding or validation for loop with AUC computation for the link prediction task
//! 
//! 1. **Embedding mode**
//! 
//! There are 2 types of embedding : hope or sketching see related docs [Hope], [NodeSketch] or [NodeSketchAsym]
//! 
//! The hope embedding requires the *hope* subcommand, the embedding relying on sketching is announced by the *sketching* subcommand.  
//! 
//! - Hope embedding can run in 2 approximations mode with a precision mode or a rank target approximation of the similatity matrix
//! 
//! - The sketching by default is adapted to the symetry declared for the csv file. It is possible to run with NodeSketchAsym on a symetric file 
//! to see the impact on validation for example.
//! 
//! example usage:
//! 
//! Hope mode for embedding with Adamic Adar approximation using approximation with a target rank of 100 and 10 iterations
//! in the range approximations:  
//! embed --csv "p2p-Gnutella09.txt" --symetric "true" embedding hope  --approx "ADA" rank --targetrank 100 --nbiter 10 
//! 
//! with precision target:  
//! embed --csv "p2p-Gnutella08.txt" --symetric "true" embedding hope  --approx "ADA" precision --epsil 0.2 --maxrank 1000 --blockiter 3
//! 
//! Sketching embedding with 3 hop neighbourhood, weight decay factor of 0.1 at each hop, dimension 500 :
//! 
//! embed --csv "p2p-Gnutella08.txt"  --symetric "true" embedding sketching --decay 0.1  --dim 500 --nbiter 3 
//! 
//! 
//! 2. **Validation mode with estimation of AUC on link prediction task**.
//! 
//!  The validation command has 2 parameters:
//! - --nbpass 
//!     It determines the number of validation pass to be done.  
//! - --skip
//!     It determines the number of edges to delete in the validation pass. Recall that the real number of discarded edges
//!     can be smaller as we must not make isolated points.
//! 
//!     It suffices to add the command : **validation --nbpass nbpass --skip fraction** before the embedding specification.
//!     Defining nbpass as the number of step asked for in the validation and skip the fraction of edges kept out of the train dataset.
//!     We get for example :  
//!   
//!     embed --csv "p2p-Gnutella08.txt" --symetric "true" validation --npass 10 --skip 0.1 sketching --decay 0.1  --dim 300 --nbiter 3
//! 
//!
//!     embed --csv "p2p-Gnutella08.txt" --symetric "true" validation --npass 10 --skip 0.1 hope --approx ADA precision --epsil 0.2 --maxrank 200  --blockiter 3
//!
//!     embed --csv "p2p-Gnutella08.txt" --symetric "true" validation --npass 10 --skip 0.1 hope --approx ADA rank --targetrank 100  --nbiter 10
//! 
//!     embed --csv wiki-Vote.txt --symetric false validation --nbpass 20 --skip 0.15 sketching --decay 0.25 --dim 500 --nbiter 2 --symetric false
//! 
//! The module can be launched (and it is recommended) by preceding the command by setting the variable RUST_LOG to info (normal information) or debug (to get related info) 
//! as for example :  *RUST_LOG=graphembed=debug embed ....*
//! 
//! It can be launched by setting  *export RUST_LOG=graphembed::validation=trace*
//! to get the maximum info in the validation module. (it will dump huge file reporting info on each edge decision)
//! 


use log::{log_enabled};

use anyhow::{anyhow};
use clap::{Arg, ArgMatches, Command, arg};

use graphembed::prelude::*;
use sprs::{TriMatI};

use graphembed::io;

use graphembed::io::bson::*;

/// variable to be used to run tests
const _DATADIR : &str = &"/home/jpboth/Data/Graphs";

#[doc(hidden)]
fn parse_sketching(matches : &ArgMatches) -> Result<NodeSketchParams, anyhow::Error> {
    log::debug!("in parse_sketching");
    // check for potential symetric argument 
    let symetric : bool = match matches.value_of("symetric") {
        Some(str) => {
            log::debug!("got symetric arg in sketching");
            let res = str.parse::<bool>();
            if res.is_err() {
                log::error!("error decoding symetric argument for sketching");
                return Err(anyhow!("error decoding symetric argument for sketching"));
            }
            else {
                res.unwrap()
            }
        },
        _   => { 
            log::debug!("setting symetric default mode in sketching");
            true
        },
    }; // end match 

    // get embedding dimension
    let dimension = match matches.value_of("dim") {
        Some(str) => {
            let res = str.parse::<usize>();
            if res.is_ok() {
                res.unwrap()
            }
            else {
                return Err(anyhow!("error parsing dim"));
            }
        },
        _   => { return Err(anyhow!("error parsing dim")); },
    }; // end match

    // get decay
    let decay = match matches.value_of("decay") {
        Some(str) => {
            str.parse::<f64>().unwrap()
        },
        _   => { return Err(anyhow!("error parsing decay")); },
    }; // end match 

    // get nbiter
    let nb_iter = match matches.value_of("nbiter") {
        Some(str) => {
            let res = str.parse::<usize>();
            if res.is_ok() {
                res.unwrap()
            }
            else {
                return Err(anyhow!("error parsing decay"));
            }
        },
        _   => { return Err(anyhow!("error parsing decay")); },
    }; // end match nb_iter
    //
    let sketch_params = NodeSketchParams{sketch_size: dimension, decay, nb_iter, symetric, parallel : true};
    return Ok(sketch_params);
} // end of parse_sketching




#[doc(hidden)]
fn parse_hope_args(matches : &ArgMatches)  -> Result<HopeParams, anyhow::Error> {
    log::debug!("in parse_hope");
    // first get mode Katz or Rooted Page Rank
    let mut epsil : f64 = 0.;
    let mut maxrank : usize = 0;
    let mut blockiter = 0;
    let mut decay : Option<f64> = None;
    // get approximation mode
    let hope_mode = match matches.value_of("proximity") {
        Some("KATZ") => {  HopeMode::KATZ
        },
        Some("RPR")  => {  HopeMode::RPR
        },
        Some("ADA")  => {  HopeMode::ADA},
        _            => {
                            log::error!("did not get proximity used : ADA,KATZ or RPR");
                            std::process::exit(1);
        },
    };

    if let Some(str)  = matches.value_of("decay")  { 
        let res = str.parse::<f64>();
        match res {
            Ok(val) => { decay = Some(val)},
            _            => {   
                                return Err(anyhow!("could not parse Hope decay"));
                            },
        };  // end of decay match 
    }
    else { // check we have decay in non ADA mode
        match hope_mode {
            HopeMode::ADA => { decay = Some(1.);},
            _ => {
                if decay.is_none() {
                    log::error!("Hope mode requires --decay for non ADA proximity");
                    println!("Hope mode requires --decay for non ADA proximity");
                    return Err(anyhow!("Hope mode requires --decay for non ADA proximity"));
                };
            },
        };
    }
    
    
    match matches.subcommand() {

        Some(("precision", sub_m)) =>  {
            if let Some(str) = sub_m.value_of("epsil") {
                let res = str.parse::<f64>();
                match res {
                    Ok(val) => { epsil = val;},
                    _            => { return Err(anyhow!("could not parse Hope epsil"));},
                }         
            } // end of epsil
 

            // get maxrank
            if let Some(str) = sub_m.value_of("maxrank") {
                let res = str.parse::<usize>();
                match res {
                    Ok(val) => { maxrank = val;},
                    _              => { return Err(anyhow!("could not parse Hope maxrank")); },
                }
            }

            // get blockiter
            if let Some(str) = sub_m.value_of("blockiter") {
                let res = str.parse::<usize>();
                match res {
                    Ok(val) => { blockiter = val;},
                    _              => { return Err(anyhow!("could not parse Hope blockiter"));},
                }        
            }
            //
            let range = RangeApproxMode::EPSIL(RangePrecision::new(epsil, blockiter, maxrank));
            let params = HopeParams::new(hope_mode, range, decay.unwrap());
            return Ok(params);
        },  // end decoding precision arg


        Some(("rank", sub_m)) => {
            if let Some(str) = sub_m.value_of("targetrank") {
                let res = str.parse::<usize>();
                match res {
                    Ok(val) => { maxrank = val;},
                    _              => { return Err(anyhow!("could not parse Hope maxrank"));},
                }
            } // end of target rank

            // get blockiter
            if let Some(str) = sub_m.value_of("nbiter") {
                let res = str.parse::<usize>();
                match res {
                    Ok(val) => { blockiter = val ; },
                    _              => {  return Err(anyhow!("could not parse Hope blockiter")); }
                }    
            }   
            //          
            let range = RangeApproxMode::RANK(RangeRank::new(maxrank, blockiter));
            let params = HopeParams::new(hope_mode, range, decay.unwrap());
            return Ok(params);
        }, // end of decoding rank arg

        _  => {
            log::error!("could not decode hope argument, got neither precision nor rank subcommands");
            return Err(anyhow!("could not parse Hope parameters"));
        },

    }; // end match
} // end of parse_hope_args

//=======================================================================

#[doc(hidden)]
#[derive(Debug)]
struct EmbeddingParams {
    mode : EmbeddingMode,
    hope : Option<HopeParams>,
    sketching : Option<NodeSketchParams>,
} // end of struct EmbeddingParams


impl From<HopeParams> for EmbeddingParams {
    fn from(params : HopeParams) -> Self {
        EmbeddingParams{mode : EmbeddingMode::Hope, hope : Some(params), sketching:None}
    }
}

impl From<NodeSketchParams> for EmbeddingParams {
    fn from(params : NodeSketchParams) -> Self {
        EmbeddingParams{mode : EmbeddingMode::NodeSketch, hope : None, sketching: Some(params)}
    }
}

//=================================================================

#[doc(hidden)]
#[derive(Debug)]
struct ValidationCmd {
    validation_params : ValidationParams,
    embedding_params : EmbeddingParams,
} // end of struct ValidationCmd




// parsing of valdation command
#[doc(hidden)]
fn parse_validation_cmd(matches : &ArgMatches) ->  Result<ValidationCmd, anyhow::Error> {
    //
    log::debug!("in parse_validation_cmd");
    // for now only link prediction is implemented
    let delete_proba : f64;
    let nbpass : usize;

    match matches.value_of("skip") {
        Some(str) =>  { 
                let res = str.parse::<f64>();
                match res {
                    Ok(val) => { delete_proba = val},
                    _       => { return Err(anyhow!("could not parse skip parameter"));
                                },
                } 
        } 
        _      => { return Err(anyhow!("could not parse decay"));}
    };  // end of skip match 

    match matches.value_of("nbpass") {
        Some(str) =>  { 
                let res = str.parse::<usize>();
                match res {
                    Ok(val) => { nbpass = val},
                    _       => { return Err(anyhow!("could not parse nbpass parameter"));
                                },
                } 
        } 
        _      => { return Err(anyhow!("could not parse decay"));}
    };  // end of skip match 
    // 
    let symetric = true; // default is symetric, we do not have here the global io parameter
    let validation_params = ValidationParams::new(delete_proba, nbpass, symetric);
    //
    let embedding_params_res = parse_embedding_cmd(matches);
    if embedding_params_res.is_ok() {
        return Ok(ValidationCmd{validation_params, embedding_params : embedding_params_res.unwrap()});
    }
    else {
        log::info!("parse_embedding_cmd failed");
        return Err(anyhow!("parse_validation_cmd parsing embedding cmd failed"));   
    }
    //
}  // end of parse_validation_cmd




// parsing of embedding command
#[doc(hidden)]
fn parse_embedding_cmd(matches : &ArgMatches) ->  Result<EmbeddingParams, anyhow::Error> {
    log::debug!("in parse_embedding_cmd");
    match matches.subcommand() {
        Some(("hope", sub_m))       => {
                if let Ok(params) = parse_hope_args(sub_m) {
                    return Ok(EmbeddingParams::from(params));
                }
                else { 
                    log::error!("parse_hope_args failed");
                    return Err(anyhow!("parse_hope_args failed"));
                }
        },
        Some(("sketching" , sub_m)) => {
                if let Ok(params) = parse_sketching(sub_m) {
                    return Ok(EmbeddingParams::from(params));
                }
                else { 
                    log::error!("parse_hope_args failed");
                    return Err(anyhow!("parse_hope_args failed"));
                }
        },
           _                                    => {
                log::error!("did not find hope neither sketching commands");
                return Err(anyhow!("parse_hope_args failed")); 
        },
    }
}  // parse_embedding_cmd

//==============================================================================================


#[doc(hidden)]
pub fn main() {
    //
    println!("initializing default logger from environment ...");
    let _ = env_logger::Builder::from_default_env().init();
    log::info!("logger initialized from default environment"); 
    //
    // first we define subcommands we will need, hope , sketching , validation
    // the hope command
    let hope_cmd = Command::new("hope")
    .subcommand_required(false)
    .arg_required_else_help(true)
    .arg(Arg::new("proximity")
        .long("approx")
        .required(true)
        .takes_value(true)
        .help("specify ADA or RPR"))
    .arg(Arg::new("decay")
        .long("decay").takes_value(true).help("RPR option needs a decay"))
    .subcommand(Command::new("precision")
        .arg_required_else_help(true)
        .args(&[
            arg!(--maxrank <maxrank> "maximum rank expected"),
            arg!(--blockiter <blockiter> "integer between 2 and 5"),
            arg!(-e --epsil <epsil> "precision between 0. and 1."),
        ]))
    .subcommand(Command::new("rank")
        .arg_required_else_help(true)
        .args(&[
            arg!(--targetrank <targetrank> "expected rank"),
            arg!(--nbiter <nbiter> "integer between 2 and 5"),
        ])          
    );

    // the sketch embedding command
    let sketch_cmd = Command::new("sketching")
        .arg(Arg::new("symetric").required(false).takes_value(true).long("symetric").help("true or false"))
        .arg_required_else_help(true)
        .args(&[
            arg!(-d --dim <dim> "the embedding dimension"),
            arg!(--decay <decay> "decay coefficient"),
            arg!(--nbiter <nbiter> "number of loops around a node"),
        ]
    );

    // validation must have one embedding subcommand
    let validation_cmd= Command::new("validation")
        .subcommand_required(true)
        .args(&[
            arg!(--nbpass <nbpass> "number of passes of validation"),
            arg!(--skip <fraction> "fraction of edges to skip in training set"),
            ])
        .subcommand(hope_cmd.clone())
        .subcommand(sketch_cmd.clone());

    // the embedding command does just the embedding
    let embedding_command = Command::new("embedding")
        .subcommand_required(true)
        .subcommand(hope_cmd.clone())
        .subcommand(sketch_cmd.clone());
    //
    // Now the command line
    // ===================
    //
    let matches = Command::new("embed")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .arg(Arg::new("csvfile")
            .long("csv")    
            .takes_value(true)
            .required(true)
            .help("expecting a csv file"))
        .arg(Arg::new("symetry")
            .short('s').long("symetric").required(true).default_value("yes")
            .help(" -s for a symetric embedding, default is symetric"))
        .arg(Arg::new("output")
            .long("output")
            .short('o')
            .takes_value(true)
            .help("-o fname for a dump in fname.bson"))
        .subcommand(embedding_command)
        .subcommand(validation_cmd)
    .get_matches();

    //
    // decode args
    // ==========

    let mut fname = String::from("");
    if matches.is_present("csvfile") {
        let csv_file = matches.value_of("csvfile").ok_or("").unwrap().parse::<String>().unwrap();
        if csv_file == "" {
            println!("parsing of request_dir failed");
            std::process::exit(1);
        }
        else {
            log::info!("input file : {:?}", csv_file.clone());
            fname = csv_file.clone();
        }
    }
    
    let symetric_graph =  match matches.value_of("symetry") {
        Some(str) => {
            let res = str.parse::<bool>();
            if res.is_ok() {
                res.unwrap()
            }
            else {
                println!("error parsing symetric , must be \"true\" or \"false\" ");
                std::process::exit(1);
            }
        },
        _   => {  // default is true
            true 
        },
    }; // end match symetry

    let mut output_params : io::output::Output = io::output::Output::default();
    if matches.is_present("output") {
        log::debug!("got output directive");
        let bson_name = matches.value_of("output").ok_or("").unwrap().parse::<String>().unwrap();
        if bson_name == "" {
            println!("parsing of bson output name failed");
            log::error!("parsing of bson output name failed");
            std::process::exit(1);
        }
        else {
            log::info!("output file : {:?}", bson_name.clone());
            let bson_output_name = Some(bson_name);
            output_params = io::output::Output::new(graphembed::io::output::Format::BSON, true, &bson_output_name);
        }
    } //end match output bson

    // now we have datafile and symetry we can parse subcommands and parameters
    let mut embedding_parameters : Option<EmbeddingParams> = None;
    let mut validation_params : Option<ValidationParams> = None;
    //
    match matches.subcommand() {
        Some(("validation", sub_m)) => {
            log::debug!("got validation command");
            let res = parse_validation_cmd(sub_m);
            match res {
                Ok(cmd) =>  { 
                    validation_params = Some(cmd.validation_params);
                    embedding_parameters = Some(cmd.embedding_params);
                },
                _                     => {  },
            }
        },

        Some(("embedding", sub_m )) => {
            log::debug!("got embedding command");
            let res = parse_embedding_cmd(sub_m);
            match res {
                Ok(params) => { embedding_parameters = Some(params); },
                _                     => { 
                    log::error!("exiting with error {}", res.err().unwrap());
                    std::process::exit(1);
                 },
            }
        }

        _  => {
            log::error!("expected subcommand hope or nodesketch");
            std::process::exit(1);
        }
    }  // end match subcommand

    if let Some(validation_m) = matches.subcommand_matches("validation") {
        log::debug!("subcommand_matches got validation");
        let res = parse_validation_cmd(validation_m);        
        match res {
            Ok(cmd) => { validation_params = Some(cmd.validation_params); },
            _                     => { 
                                    log::error!("paring validation command failed");
                                    println!("exiting with error {}", res.err().as_ref().unwrap());
                                  //  log::error!("exiting with error {}", res.err().unwrap());
                                    std::process::exit(1);                                
            },
        }
    }  // end if validation

    // examine embedding_parameters to see if we do hope or sketching
    
    log::info!(" parsing of commands succeeded"); 
    log::debug!("\n embedding paramertes : {:?}", embedding_parameters.as_ref());
    //
//    let path = std::path::Path::new(_DATADIR).join(fname.clone());
    let path = std::path::Path::new(&fname);
    if log_enabled!(log::Level::Info) {
        log::info!("\n\n loading file {:?}, symetric = {}", path, symetric_graph);
    }
    else {
        println!("\n\n loading file {:?}, symetric = {}", path, symetric_graph);
    }
    // TODO change argument directed to symetric to csv_to_trimat_delimiters to avoid the !
    let res = csv_to_trimat_delimiters::<f64>(&path, !symetric_graph);
    if res.is_err() {
        log::error!("error : {:?}", res.as_ref().err());
        log::error!("embedder failed in csv_to_trimat, reading {:?}", &path);
        std::process::exit(1);
    }
    let (trimat, node_index) = res.unwrap();
    //
    // we have our graph in trimat format, we must pass info on symetry or asymetry
    //
    let embedding_parameters = embedding_parameters.unwrap();

    match embedding_parameters.mode  {
        EmbeddingMode::Hope => {
            log::info!("embedding mode : Hope");
            log::debug!(" hope parameters : {:?}",embedding_parameters.hope.unwrap());
            // now we allocate an embedder (sthing that implement the Embedder trait)
            if validation_params.is_none() {
                // we do the embedding
                let mut hope = Hope::new(embedding_parameters.hope.unwrap(), trimat); 
                let embedding = Embedding::new(node_index, &mut hope);
                if embedding.is_err() {
                    log::error!("hope embedding failed, error : {:?}", embedding.as_ref().err());
                    std::process::exit(1);
                };
                let embed_res = embedding.unwrap();
                // should dump somewhere
                let res = bson_dump(&embed_res, &output_params);
                if res.is_err() {
                    log::error!("bson dump in {} failed", output_params.get_output_name());
                }
            }
            else  {
                let mut params = validation_params.unwrap();
                if !symetric_graph {
                    log::debug!("setting asymetric flag for validation");
                    params = ValidationParams::new(params.get_delete_fraction(), params.get_nbpass(), symetric_graph);
                }
                log::debug!("validation parameters : {:?}", params);
                // have to run validation simulations
                log::info!("doing validaton runs for hope embedding");
                // construction of the function necessay for AUC iterations
                let f = | trimat : TriMatI<f64, usize> | -> EmbeddedAsym<f64> {
                    let mut hope = Hope::new(embedding_parameters.hope.unwrap(), trimat); 
                    let res = hope.embed();
                    res.unwrap()
                };
                link::estimate_auc(&trimat.to_csr(), params.get_nbpass(), params.get_delete_fraction(), symetric_graph, &f);
            }
        },  // end case Hope

        EmbeddingMode::NodeSketch => {
            log::info!("embedding mode : Sketching");
            let sketching_params = embedding_parameters.sketching.unwrap();
            // check coherence with symetry of graph, symetric graph can run in asymetric mode as reading csv we symetrize, the inverse not possible
            if !symetric_graph && sketching_params.is_symetric(){
                log::info!("asymetric graph requires asymetric embedding, use --symetry false in sketching command");
                println!("asymetric graph requires asymetric embedding, use --symetry false in sketching command");
                std::process::exit(1);
            }
            else if symetric_graph && !sketching_params.is_symetric() {
                log::info!("doing asymetric embedding for symetric graph");
            }          
            log::debug!(" sketching embedding parameters : {:?}", sketching_params);
            if validation_params.is_none() {
                log::debug!("running embedding without validation");
                let sketching_symetry = sketching_params.is_symetric();
                // now we allocate an embedder (sthing that implement the Embedder trait)
                match sketching_symetry {
                    true => {   
                        let mut nodesketch = NodeSketch::new(sketching_params, trimat);
                        let embedding = Embedding::new(node_index, &mut nodesketch);
                        if embedding.is_err() {
                            log::error!("nodesketch embedding failed error : {:?}", embedding.as_ref().err());
                            std::process::exit(1);
                        };
                        let embed_res = embedding.unwrap();             
                        let res = bson_dump(&embed_res, &output_params);
                        if res.is_err() {
                            log::error!("bson dump in {} failed", output_params.get_output_name());
                        }   
                    },
                    false => {  
                        let mut nodesketchasym = NodeSketchAsym::new(embedding_parameters.sketching.unwrap(), trimat);
                        let embedding = Embedding::new(node_index, &mut nodesketchasym);
                        if embedding.is_err() {
                            log::error!("nodesketchasym embedding failed error : {:?}", embedding.as_ref().err());
                            std::process::exit(1);
                        };
                        let embed_res = embedding.unwrap();
                        // should dump somewhere
                        let res = bson_dump(&embed_res, &output_params);
                        if res.is_err() {
                            log::error!("bson dump in {} failed", output_params.get_output_name());
                        }    
                    },  // end asymetric sketching
                };
            } // end case no validation
            else {
                let validation_params = validation_params.unwrap();
                log::debug!("validation , validation parameters : {:?}", validation_params);
                log::debug!("sketching parameters : {:?}", sketching_params);
                let sketching_symetry = sketching_params.is_symetric();
                if !symetric_graph && sketching_params.is_symetric() {
                    log::info!("asymetric graph requires asymetric embedding, use --symetry false in sketching command");
                    println!("asymetric graph requires asymetric embedding, use --symetry false in sketching command");
                    std::process::exit(1);
                }
                // we are sure to have a coherent symetry arguments
                if !sketching_symetry {
                    log::info!("doing validaton runs for nodesketchasym embedding");
                    // construction of the function necessay for AUC iterations            
                    let f = | trimat : TriMatI<f64, usize> | -> EmbeddedAsym<usize> {
                        let mut nodesketch = NodeSketchAsym::new(embedding_parameters.sketching.unwrap(), trimat);
                        let res = nodesketch.embed();
                        res.unwrap()
                    };
                    link::estimate_auc(&trimat.to_csr(), validation_params.get_nbpass(), validation_params.get_delete_fraction(), symetric_graph, &f);
                }  // end case asymetric
                else {
                    log::info!("doing validaton runs for nodesketch embedding");
                    let f = | trimat : TriMatI<f64, usize> | -> Embedded<usize> {
                        let mut nodesketch = NodeSketch::new(sketching_params, trimat);
                        let res = nodesketch.embed();
                        res.unwrap()
                    };
                    link::estimate_auc(&trimat.to_csr(), validation_params.get_nbpass(), validation_params.get_delete_fraction(), symetric_graph, &f);
                }
                // TODO precision estimation too costly must subsample
                //    estimate_precision(&trimat.to_csr(), params.get_nbpass(), params.get_delete_fraction(), false, &f);
            }
        },  // end case sketching_params
    };
    // 
    //    
}  // end fo main