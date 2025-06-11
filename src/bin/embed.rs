//! an executable for embedding graph
//!
//! The main arguments are
//!  - --csv filename
//!  - --symetric "true"  if present specifies the graph is symetric. In this case the csv file describes a symetric (half of the edges in csv) or asymetric graph.  
//!    If the file is declared symetric, each edge(a,b) is completed upon reading by the edge (b,a).  
//!    Sometimes a symetric graph is fully described in the csv file, then declare the file as asymetric (--symetric "false").
//!    Then graph is deduced only from the read edges.  
//!    **In this case for the validation parameters by sketching (which can do validation in symetric or asymetric mode) the flag --symetric is required to get deletion of both edges i->j and j->i**
//!     
//!  - --output or -o filename  
//!    This dumps the embedding in a bson file named filename.bson. See module [bson].  
//!    By default an embedding is written in the file **embedding.bson**.
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
//!   to see the impact on validation for example.
//!
//! example usage:
//!
//! Hope mode for embedding with Adamic Adar approximation using approximation with a target rank of 100 and 10 iterations
//! in the range approximations:  
//! embed --csv "p2p-Gnutella09.txt" --symetric "true"  embedding hope  rank --targetrank 100 --nbiter 10 --output outputname
//!
//! with precision target:  
//! embed --csv "p2p-Gnutella08.txt" --symetric "true" embedding hope   precision --epsil 0.2 --maxrank 1000 --blockiter 3 --output outputname
//!
//! Sketching embedding with 3 hop neighbourhood, weight decay factor of 0.1 at each hop, dimension 500 :
//!
//! embed --csv "p2p-Gnutella08.txt"  --symetric "true" embedding sketching --decay 0.1  --dim 500 --nbiter 3 --output outputname
//!
//!
//! 2. **Validation mode with estimation of AUC on link prediction task**.
//!
//!  The validation command has 2 parameters and 2 possible flaga:
//! - --nbpass  
//!   It determines the number of validation pass to be done.  
//! - --skip  
//!   It determines the number of edges to delete in the validation pass. Recall that the real number of discarded edges
//!   can be smaller as we must not make isolated points.
//! - --centric  
//!   This flag is optional and asks for one pass of centric AUC computation after standard AUC link prediction (See [graphembed::validation::link::estimate_centric_auc()])
//!
//!
//!     embed --csv "p2p-Gnutella08.txt" --symetric "true" validation [--centric] --nbpass 10 --skip 0.1 hope  precision --epsil 0.2 --maxrank 200  --blockiter 3
//!
//!     embed --csv "p2p-Gnutella08.txt" --symetric "true" validation [--centric] --nbpass 10 --skip 0.1 hope  rank --targetrank 100  --nbiter 10
//!
//!     embed --csv wiki-Vote.txt --symetric false validation [--centric] --nbpass 20 --skip 0.15 sketching --decay 0.25 --dim 500 --nbiter 2
//!
//! The module can be launched (and it is recommended) by preceding the command by setting the variable RUST_LOG to info (normal information) or debug (to get related info)
//! as for example :  *RUST_LOG=graphembed=debug embed ....*
//!
//! It can be launched by setting  *export RUST_LOG=graphembed::validation=trace*
//! to get the maximum info in the validation module. (it will dump huge file reporting info on each edge decision)
//!

use log::log_enabled;

use anyhow::anyhow;
use clap::{Arg, ArgAction, ArgMatches, Command};

use graphembed::prelude::*;
use sprs::TriMatI;

use graphembed::io;

/// variable to be used to run tests
const _DATADIR: &str = "/home/jpboth/Data/Graph";

#[doc(hidden)]
fn parse_sketching(
    matches: &ArgMatches,
    symetric: bool,
) -> Result<NodeSketchParams, anyhow::Error> {
    log::debug!("in parse_sketching");
    //
    let dimension = *matches
        .get_one::<usize>("dimension")
        .expect("dim value required");
    let decay = *matches
        .get_one::<f64>("decay")
        .expect("decay float value required");
    let nb_iter = *matches
        .get_one::<usize>("nbiter")
        .expect("nb_iter value required");
    // we get symetry from upper args
    let sketch_params = NodeSketchParams {
        sketch_size: dimension,
        decay,
        nb_iter,
        symetric,
        parallel: true,
    };
    Ok(sketch_params)
} // end of parse_sketching

#[doc(hidden)]
fn parse_hope_args(matches: &ArgMatches) -> Result<HopeParams, anyhow::Error> {
    log::debug!("in parse_hope");
    // first get mode Katz or Rooted Page Rank
    let hope_mode = HopeMode::ADA; // only ADA for now
    let decay = 1.;
    match matches.subcommand() {
        Some(("precision", sub_m)) => {
            let epsil = *sub_m
                .get_one::<f64>("epsil")
                .expect("could not parse Hope epsil");
            let maxrank = *sub_m
                .get_one::<usize>("maxrank")
                .expect("could not parse Hope maxrank");
            let blockiter = *sub_m
                .get_one::<usize>("blockiter")
                .expect("could not parse Hope blockiter");
            //
            let range = RangeApproxMode::EPSIL(RangePrecision::new(epsil, blockiter, maxrank));
            let params = HopeParams::new(hope_mode, range, decay);
            Ok(params)
        } // end decoding precision arg

        Some(("rank", sub_m)) => {
            let targetrank = *sub_m
                .get_one::<usize>("targetrank")
                .expect("could not parse Hope target rank");
            let blockiter = *sub_m
                .get_one::<usize>("nbiter")
                .expect("could not parse Hope nbiter");
            //
            let range = RangeApproxMode::RANK(RangeRank::new(targetrank, blockiter));
            let params = HopeParams::new(hope_mode, range, decay);
            Ok(params)
        } // end of decoding rank arg

        _ => {
            log::error!(
                "could not decode hope argument, got neither precision nor rank subcommands"
            );
            Err(anyhow!("could not parse Hope parameters"))
        }
    } // end match
} // end of parse_hope_args

//=======================================================================

#[doc(hidden)]
#[derive(Debug)]
struct EmbeddingParams {
    mode: EmbeddingMode,
    hope: Option<HopeParams>,
    sketching: Option<NodeSketchParams>,
} // end of struct EmbeddingParams

impl From<HopeParams> for EmbeddingParams {
    fn from(params: HopeParams) -> Self {
        EmbeddingParams {
            mode: EmbeddingMode::Hope,
            hope: Some(params),
            sketching: None,
        }
    }
}

impl From<NodeSketchParams> for EmbeddingParams {
    fn from(params: NodeSketchParams) -> Self {
        EmbeddingParams {
            mode: EmbeddingMode::NodeSketch,
            hope: None,
            sketching: Some(params),
        }
    }
}

//=================================================================

#[doc(hidden)]
#[derive(Debug)]
struct ValidationCmd {
    validation_params: ValidationParams,
    embedding_params: EmbeddingParams,
} // end of struct ValidationCmd

// parsing of valdation command
#[doc(hidden)]
fn parse_validation_cmd(
    matches: &ArgMatches,
    symetric: bool,
) -> Result<ValidationCmd, anyhow::Error> {
    //
    log::debug!("in parse_validation_cmd");
    // for now only link prediction is implemented
    let nbpass = *matches
        .get_one::<usize>("nbpass")
        .expect("number of validation pass required");
    //
    let delete_proba = *matches
        .get_one::<f64>("skip")
        .expect("could not parse skip parameter");
    //
    let centric: bool = matches.get_flag("centric");
    if centric {
        log::info!("doing a centric auc pass after standard AUC link prediction");
    } else {
        log::info!("no centric pass on link prediction");
    }
    //
    let validation_params = ValidationParams::new(delete_proba, nbpass, symetric, centric);
    //
    let embedding_cmd_res = parse_embedding_cmd(matches, symetric);

    if embedding_cmd_res.is_ok() {
        // for validation we do not need ouput
        let embedding_cmd = embedding_cmd_res.unwrap();
        Ok(ValidationCmd {
            validation_params,
            embedding_params: embedding_cmd.0,
        })
    } else {
        log::info!("parse_embedding_cmd failed");
        Err(anyhow!("parse_validation_cmd parsing embedding cmd failed"))
    }
    //
} // end of parse_validation_cmd

// parsing of embedding command
#[doc(hidden)]
fn parse_embedding_cmd(
    matches: &ArgMatches,
    symetric_graph: bool,
) -> Result<(EmbeddingParams, io::output::Output), anyhow::Error> {
    log::debug!("in parse_embedding_cmd");
    //
    let bson_output_name = matches.get_one::<String>("output");
    let output_name: Option<String>;

    if bson_output_name.is_some() {
        output_name = Some(bson_output_name.unwrap().clone());
        log::info!("will ouput embedding in bson file :  {:?}", output_name);
    } else {
        output_name = None;
    }
    //
    let output_params =
        io::output::Output::new(graphembed::io::output::Format::BSON, true, &output_name);
    //
    match matches.subcommand() {
        Some(("hope", sub_m)) => {
            if let Ok(params) = parse_hope_args(sub_m) {
                Ok((EmbeddingParams::from(params), output_params))
            } else {
                log::error!("parse_hope_args failed");
                Err(anyhow!("parse_hope_args failed"))
            }
        }
        Some(("sketching", sub_m)) => {
            if let Ok(params) = parse_sketching(sub_m, symetric_graph) {
                Ok((EmbeddingParams::from(params), output_params))
            } else {
                log::error!("parse_sketching failed");
                Err(anyhow!("parse_sketching failed"))
            }
        }
        _ => {
            log::error!("did not find hope neither sketching commands");
            Err(anyhow!("parse_hope_args failed"))
        }
    }
} // parse_embedding_cmd

//==============================================================================================

#[doc(hidden)]
pub fn main() {
    // TODO: to put in clap ? just for now select at compile time
    let do_vcmpr = true;
    //
    println!("initializing default logger from environment ...");
    env_logger::Builder::from_default_env().init();
    log::info!("logger initialized from default environment");
    //
    // first we define subcommands we will need, hope , sketching , validation
    // the hope command
    //
    let hope_cmd = Command::new("hope")
        .subcommand_required(false)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("precision")
                .arg_required_else_help(true)
                .arg(
                    Arg::new("epsil")
                        .long("epsil")
                        .help("precision between 0. and 1.")
                        .required(true)
                        .action(ArgAction::Set)
                        .value_parser(clap::value_parser!(f64)),
                )
                .arg(
                    Arg::new("maxrank")
                        .long("maxrank")
                        .help("maximum rank expected")
                        .required(true)
                        .action(ArgAction::Set)
                        .value_parser(clap::value_parser!(usize)),
                )
                .arg(
                    Arg::new("blockiter")
                        .long("blockiter")
                        .help("blockiter")
                        .required(true)
                        .help("integer between 2 and 5")
                        .action(ArgAction::Set)
                        .value_parser(clap::value_parser!(usize)),
                ),
        )
        .subcommand(
            Command::new("rank")
                .arg_required_else_help(true)
                .arg(
                    Arg::new("targetrank")
                        .help("rank expected")
                        .long("targetrank")
                        .required(true)
                        .action(ArgAction::Set)
                        .value_parser(clap::value_parser!(usize)),
                )
                .arg(
                    Arg::new("nbiter")
                        .long("nbiter")
                        .help("number of iterations")
                        .required(true)
                        .help("integer between 2 and 5")
                        .action(ArgAction::Set)
                        .value_parser(clap::value_parser!(usize)),
                ),
        );
    // the sketch embedding command
    let sketch_cmd = Command::new("sketching")
        .arg(
            Arg::new("dimension")
                .required(true)
                .short('d')
                .long("dim")
                .help("the embedding dimension")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("decay")
                .required(true)
                .long("decay")
                .help("decay coefficient")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("nbiter")
                .required(true)
                .long("nbiter")
                .help("number of loops around a node ")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize)),
        );

    // validation must have one embedding subcommand
    let validation_cmd = Command::new("validation")
        .subcommand_required(true)
        .arg(
            Arg::new("nbpass")
                .required(true)
                .long("nbpass")
                .help("number  of passes of validation")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("skip")
                .required(true)
                .long("skip")
                .help("fraction of edges to skip in training set")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("centric")          // do we process amino acid file, default is dna, pass --aa
               .long("centric")
               .action(clap::ArgAction::SetTrue)
                .help("--centric To ask for a centric validation pass after standard one, require no value")
            )
        .subcommand(hope_cmd.clone())
        .subcommand(sketch_cmd.clone());

    // the embedding command does just the embedding
    let embedding_command = Command::new("embedding")
        .arg(
            Arg::new("output")
                .long("output")
                .short('o')
                .value_parser(clap::value_parser!(String))
                .action(ArgAction::Set)
                .help("-o fname for a dump in fname.bson"),
        )
        .subcommand_required(true)
        .subcommand(hope_cmd.clone())
        .subcommand(sketch_cmd.clone());
    //
    // Now the command line
    // ===================
    //
    let matches = Command::new("embed")
        .arg_required_else_help(true)
        .arg(
            Arg::new("csvfile")
                .long("csv")
                .required(true)
                .value_parser(clap::value_parser!(String))
                .action(ArgAction::Set)
                .help("expecting a csv file"),
        )
        .arg(
            Arg::new("symetry")
                .short('s')
                .long("symetric")
                .required(true)
                .value_parser(clap::value_parser!(bool))
                .default_value("true")
                .help(" -s "),
        )
        .subcommand_required(true)
        .subcommand(embedding_command)
        .subcommand(validation_cmd)
        .get_matches();

    //
    // decode args
    // ==========

    let fname = matches
        .get_one::<String>("csvfile")
        .expect("need a csv file");

    let symetric_graph = *matches.get_one::<bool>("symetry").expect("true or false");

    // now we have datafile and symetry we can parse subcommands and parameters
    let embedding_parameters: Option<EmbeddingParams>;
    let mut validation_params: Option<ValidationParams> = None;
    let mut output_params: Option<io::output::Output> = None;
    //
    match matches.subcommand() {
        Some(("validation", sub_m)) => {
            log::debug!("got validation command");
            let res = parse_validation_cmd(sub_m, symetric_graph);
            match res {
                Ok(cmd) => {
                    validation_params = Some(cmd.validation_params);
                    embedding_parameters = Some(cmd.embedding_params);
                }
                _ => {
                    log::error!(
                        "exiting with error in parsing validation command{}",
                        res.err().unwrap()
                    );
                    std::process::exit(1);
                }
            }
        }

        Some(("embedding", sub_m)) => {
            log::debug!("got embedding command");
            let res = parse_embedding_cmd(sub_m, symetric_graph);
            match res {
                Ok((embedding_params, embed_ouput_params)) => {
                    embedding_parameters = Some(embedding_params);
                    output_params = Some(embed_ouput_params);
                }
                _ => {
                    log::error!("exiting with error {}", res.err().unwrap());
                    std::process::exit(1);
                }
            }
        }

        _ => {
            log::error!("expected subcommand hope or nodesketch");
            std::process::exit(1);
        }
    } // end match subcommand

    if let Some(validation_m) = matches.subcommand_matches("validation") {
        log::debug!("subcommand_matches got validation");
        let res = parse_validation_cmd(validation_m, symetric_graph);
        match res {
            Ok(cmd) => {
                validation_params = Some(cmd.validation_params);
            }
            _ => {
                log::error!("paring validation command failed");
                println!("exiting with error {}", res.err().as_ref().unwrap());
                //  log::error!("exiting with error {}", res.err().unwrap());
                std::process::exit(1);
            }
        }
    } // end if validation

    // examine embedding_parameters to see if we do hope or sketching
    // embedding directive without validation and no dump is most probably an error
    if embedding_parameters.is_some() && validation_params.is_none() {
        if output_params.is_none() {
            log::error!(
                "embedding asked without validation and no output given to dump the embedding ..., are you sure"
            );
            std::process::exit(1);
        }
    }
    log::info!(" parsing of commands succeeded");
    log::debug!(
        "\n embedding paramertes : {:?}",
        embedding_parameters.as_ref()
    );
    //
    let path = std::path::Path::new(fname);
    if !path.exists() {
        log::error!("file do not exist : {:?}", fname);
        println!("file do not exist : {:?}", fname);
        std::process::exit(1);
    }
    if log_enabled!(log::Level::Info) {
        log::info!(
            "\n\n loading file {:?}, symetric = {}",
            path,
            symetric_graph
        );
    } else {
        println!(
            "\n\n loading file {:?}, symetric = {}",
            path, symetric_graph
        );
    }
    // TODO change argument directed to symetric to csv_to_trimat_delimiters to avoid the !
    let res = csv_to_trimat_delimiters::<f64>(path, !symetric_graph);
    if res.is_err() {
        log::error!("error : {:?}", res.as_ref().err());
        log::error!("embedder failed in csv_to_trimat, reading {:?}", path);
        std::process::exit(1);
    }
    let (trimat, node_index) = res.unwrap();
    //
    // we have our graph in trimat format, we must pass info on symetry or asymetry
    //
    let embedding_parameters = embedding_parameters.unwrap();

    match embedding_parameters.mode {
        EmbeddingMode::Hope => {
            log::info!("embedding mode : Hope");
            log::info!(
                " hope parameters : {:?}",
                embedding_parameters.hope.unwrap()
            );
            // now we allocate an embedder (sthing that implement the Embedder trait)
            if validation_params.is_none() {
                // we do the embedding
                let mut hope = Hope::new(embedding_parameters.hope.unwrap(), trimat);
                let embedding = Embedding::new(node_index, &mut hope);
                if embedding.is_err() {
                    log::error!(
                        "hope embedding failed, error : {:?}",
                        embedding.as_ref().err()
                    );
                    std::process::exit(1);
                };
                let embed_res = embedding.unwrap();
                // should dump somewhere
                let output = output_params.as_ref().unwrap();
                let res = bson_dump(&embed_res, output);
                if res.is_err() {
                    log::error!("bson dump in {} failed", output.get_output_name());
                }
            } else {
                let mut params = validation_params.unwrap();
                if !symetric_graph {
                    log::info!("asymetric graph, setting asymetric flag for validation");
                    params = ValidationParams::new(
                        params.get_delete_fraction(),
                        params.get_nbpass(),
                        symetric_graph,
                        params.do_centric(),
                    );
                }
                log::debug!("validation parameters : {:?}", params);
                // have to run validation simulations
                log::info!("doing validaton runs for hope embedding");
                // construction of the function necessay for AUC iterations
                let f = |trimat: TriMatI<f64, usize>| -> EmbeddedAsym<f64> {
                    let mut hope = Hope::new(embedding_parameters.hope.unwrap(), trimat);
                    let res = hope.embed();
                    res.unwrap()
                };
                link::estimate_auc(
                    &trimat.to_csr(),
                    params.get_nbpass(),
                    params.get_delete_fraction(),
                    symetric_graph,
                    &f,
                );
                if params.do_centric() {
                    if do_vcmpr {
                        /* To costly
                        // if vcmpr is asked we produce also standard precision and recall for comparison
                        link::estimate_precision(
                            &trimat.to_csr(),
                            params.get_nbpass(),
                            params.get_delete_fraction(),
                            symetric_graph,
                            &f,
                        );
                        */
                        //
                        link::estimate_vcmpr(
                            &trimat.to_csr(),
                            params.get_nbpass(),
                            10,
                            params.get_delete_fraction(),
                            symetric_graph,
                            &f,
                        );
                    }
                    //
                    link::estimate_centric_auc(
                        &trimat.to_csr(),
                        params.get_nbpass(),
                        params.get_delete_fraction(),
                        symetric_graph,
                        &f,
                    );
                }
            }
        } // end case Hope

        EmbeddingMode::NodeSketch => {
            log::info!("embedding mode : Sketching");
            let sketching_params = embedding_parameters.sketching.unwrap();
            log::info!(" sketching embedding parameters : {:?}", sketching_params);
            if validation_params.is_none() {
                log::debug!("running embedding without validation");
                let sketching_symetry = sketching_params.is_symetric();
                // now we allocate an embedder (sthing that implement the Embedder trait)
                match sketching_symetry {
                    true => {
                        let mut nodesketch = NodeSketch::new(sketching_params, trimat);
                        let embedding = Embedding::new(node_index, &mut nodesketch);
                        if embedding.is_err() {
                            log::error!(
                                "nodesketch embedding failed error : {:?}",
                                embedding.as_ref().err()
                            );
                            std::process::exit(1);
                        };
                        let embed_res = embedding.unwrap();
                        let output = output_params.as_ref().unwrap();
                        let res = bson_dump(&embed_res, output);
                        if res.is_err() {
                            log::error!("bson dump in {} failed", output.get_output_name());
                        }
                    }
                    false => {
                        let mut nodesketchasym =
                            NodeSketchAsym::new(embedding_parameters.sketching.unwrap(), trimat);
                        let embedding = Embedding::new(node_index, &mut nodesketchasym);
                        if embedding.is_err() {
                            log::error!(
                                "nodesketchasym embedding failed error : {:?}",
                                embedding.as_ref().err()
                            );
                            std::process::exit(1);
                        };
                        let embed_res = embedding.unwrap();
                        // should dump somewhere
                        let output = output_params.as_ref().unwrap();
                        let res = bson_dump(&embed_res, output);
                        if res.is_err() {
                            log::error!("bson dump in {} failed", output.get_output_name());
                        }
                    } // end asymetric sketching
                };
            }
            // end case no validation
            else {
                let validation_params = validation_params.unwrap();
                log::debug!(
                    "validation , validation parameters : {:?}",
                    validation_params
                );
                log::debug!("sketching parameters : {:?}", sketching_params);
                let sketching_symetry = sketching_params.is_symetric();
                // we are sure to have a coherent symetry arguments
                if !sketching_symetry {
                    log::info!("doing validaton runs for nodesketchasym embedding");
                    // construction of the function necessay for AUC iterations
                    let f = |trimat: TriMatI<f64, usize>| -> EmbeddedAsym<usize> {
                        let mut nodesketch =
                            NodeSketchAsym::new(embedding_parameters.sketching.unwrap(), trimat);
                        let res = nodesketch.embed();
                        res.unwrap()
                    };
                    link::estimate_auc(
                        &trimat.to_csr(),
                        validation_params.get_nbpass(),
                        validation_params.get_delete_fraction(),
                        symetric_graph,
                        &f,
                    );
                }
                // end case asymetric
                else {
                    log::info!("doing validaton runs for nodesketch embedding");
                    let f = |trimat: TriMatI<f64, usize>| -> Embedded<usize> {
                        let mut nodesketch = NodeSketch::new(sketching_params, trimat);
                        let res = nodesketch.embed();
                        res.unwrap()
                    };
                    link::estimate_auc(
                        &trimat.to_csr(),
                        validation_params.get_nbpass(),
                        validation_params.get_delete_fraction(),
                        symetric_graph,
                        &f,
                    );
                    // we compare with VCMPR
                    if validation_params.do_centric() {
                        log::info!("doing precision estimation normal and centric modes ");
                        if do_vcmpr {
                            /* To costly:
                            if vcmpr is asked we produce also standard precision and recall for comparison
                            link::estimate_precision(
                                &trimat.to_csr(),
                                2,
                                validation_params.get_delete_fraction(),
                                symetric_graph,
                                &f,
                            );
                            */
                            link::estimate_vcmpr(
                                &trimat.to_csr(),
                                2,
                                10,
                                validation_params.get_delete_fraction(),
                                symetric_graph,
                                &f,
                            );
                        }
                        //
                        link::estimate_centric_auc(
                            &trimat.to_csr(),
                            validation_params.get_nbpass(),
                            validation_params.get_delete_fraction(),
                            symetric_graph,
                            &f,
                        );
                    }
                }
                // TODO precision estimation too costly must subsample
                //    estimate_precision(&trimat.to_csr(), params.get_nbpass(), params.get_delete_fraction(), false, &f);
            }
        } // end case sketching_params
    };
    //
    //
} // end fo main
