//! an executable for embedding graph
//! example usage:
//! embedder --csv "p2p-Gnutella09.txt" hope --decay 0.1 --approx "RPR" precision --epsil 0.1 --maxrank 1000 --blockiter 3
//! embedder --csv "p2p-Gnutella09.txt" hope --decay 0.1 --approx "RPR" rank --targetrank 1000 --nbiter 3
//! embedder --csv "p2p-Gnutella09.txt" nodesketch --decay 0.1  --dim 500 --nbiter 3
//! 
//!  hope or nodesketch are differents algorithms for embedding see related docs
//!  for hope algorithms different modes of approximations are possible : KATZ, RPR (rooted page rank), ADA (adamic adar)
//!  




use anyhow::{anyhow};
use clap::{Arg, ArgMatches, Command, arg};

use graphite::prelude::*;
use crate::{nodesketch::*};
use sprs::{TriMatI};

static DATADIR : &str = &"/home/jpboth/Data/Graphs";


fn parse_sketching(matches : &ArgMatches) -> Result<NodeSketchParams, anyhow::Error> {
    log::debug!("in parse_sketching");
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
    }; // end match

    //
    let sketch_params = NodeSketchParams{sketch_size: dimension, decay, nb_iter, parallel : true};
    return Ok(sketch_params);
} // end of parse_sketching


fn parse_hope_args(matches : &ArgMatches)  -> Result<HopeParams, anyhow::Error> {
    log::debug!("in parse_hope");
    // first get mode Katz or Rooted Page Rank
    let mut epsil : f64 = 0.;
    let mut maxrank : usize = 0;
    let mut blockiter = 0;
    let decay;
    // get approximation mode
    let hope_mode = match matches.value_of("proximity") {
        Some("KATZ") => { HopeMode::KATZ},
        Some("RPR")  => { HopeMode::RPR},
        Some("ADA")  => { HopeMode::ADA},
        _            => {
                            log::error!("did not get proximity used, KATZ or RPR");
                            std::process::exit(1);
                        },
    };
    // get decay
    match matches.value_of("decay") {
        Some(str) =>  { 
                let res = str.parse::<f64>();
                match res {
                    Ok(val) => { decay = val},
                    _       => { return Err(anyhow!("could not parse Hope decay"));},
                } 
        } 
        _      => { return Err(anyhow!("could not parse decay"));}

    };                         
    
    
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
            let params = HopeParams::new(hope_mode, range, decay);
            return Ok(params);
        },  // end decoding preciison arg


        Some(("rank", sub_m)) => {
            if let Some(str) = sub_m.value_of("targetrank") {
                let res = str.parse::<usize>();
                match res {
                    Ok(val) => { maxrank = val;},
                    _              => { return Err(anyhow!("could not parse Hope maxrank"));},
                }
            }

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
            let params = HopeParams::new(hope_mode, range, decay);
            return Ok(params);
        }, // end of decoding rank arg

        _  => {
            log::error!("could not decode hope argument, got neither precision nor rank subcommands");
            return Err(anyhow!("could not parse Hope parameters"));
        },

    }; // end match
} // end of parse_hope_args



pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    log::info!("logger initialized"); 
    //
    let matches = Command::new("embed")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .arg(Arg::new("csvfile")
            .long("csv")    
            .takes_value(true)
            .required(true)
            .help("expecting a csv file"))
        .arg(Arg::new("embedder")
            .long("embedder")
            .required(false)
            .takes_value(true)
            .help("specify \"hope\" or \"sketching\" "))
        .subcommand(Command::new("hope")
            .subcommand_required(true)
            .arg_required_else_help(true)
            .arg(Arg::new("proximity")
                .long("approx")
                .required(true)
                .takes_value(true)
                .help("specify KATZ or RPR"))
            .arg(Arg::new("decay")
                .long("decay")
                .takes_value(true)
                .required(true)
                .help("give a small value between 0. and 1.")
            )
            .subcommand(Command::new("precision")
                .arg_required_else_help(true)
                .args(&[
                    arg!(--maxrank <maxrank> "maximum rank expected"),
                    arg!(--blockiter <blockiter> "integer between 2 and 5"),
                    arg!(-e --epsil <epsil> "precision between 0. and 1."),
                ])
            )
            .subcommand(Command::new("rank")
                .arg_required_else_help(true)
                .args(&[
                    arg!(--targetrank <targetrank> "expected rank"),
                    arg!(--nbiter <nbiter> "integer between 2 and 5"),
                ])
            )
        )
        .subcommand(Command::new("sketching")
            .arg_required_else_help(true)
            .args(&[
                arg!(-d --dim <dim> "the embedding dimension"),
                arg!(--decay <decay> "decay coefficient"),
                arg!(--nbiter <nbiter> "number of loops around a node"),
            ])
            .arg(Arg::new("symetry")
                .short('a')
                .help(" -a for asymetric embedding, default is symetric"))
        )
    .get_matches();

    // decode args

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
    };

    let mut hope_params : Option<HopeParams> = None;
    let mut sketching_params : Option<NodeSketchParams> = None;
    //
    match matches.subcommand() {
        Some(("hope", sub_m)) => {
            log::debug!("got hope mode");
            let res = parse_hope_args(sub_m);
            match res {
                Ok(params) => { hope_params = Some(params); },
                _                     => {  },
            }
        },

        Some(("sketching", sub_m )) => {
            log::debug!("got sketching mode");
            let res = parse_sketching(sub_m);
            match res {
                Ok(params) => { sketching_params = Some(params); },
                _                     => {  },
            }
        }

        _  => {
            log::error!("expected subcommand hope or nodesketch");
            std::process::exit(1);
        }
    }  // end match subcommand


    log::info!("in hope::test_hope_gnutella09"); 
    // Nodes: 8114 Edges: 26013
    let path = std::path::Path::new(crate::DATADIR).join(fname.clone().as_str());
    let delimiters = [b'\t', b',', b' '];
    log::info!("\n\n test_nodesketchasym_wiki, loading file {:?}", path);
    let mut res: anyhow::Result<(TriMatI<f64, usize>, NodeIndexation<usize>)> = Err(anyhow!("not initializd"));
    for delim in delimiters {
        log::info!("embedder trying reading {:?} with  delimiter{ }", &path, delim);
        res = csv_to_trimat::<f64>(&path, true, delim);
        if res.is_err() {
            log::error!("embedder failed in csv_to_trimat, reading {:?}, trying delimiter {} ", &path, delim);
        }
        else { break;}
    }
    if res.is_err() {
        log::error!("error : {:?}", res.as_ref().err());
        log::error!("embedder failed in csv_to_trimat, reading {:?}", &path);
        std::process::exit(1);
    }
    let (trimat, node_index) = res.unwrap();
    //
    // we have our graph in trimat format
    //
    if hope_params.is_some() {
        log::info!("embedding mode : Hope");
        // now we embed
        let mut hope = Hope::new(hope_params.unwrap(), trimat); 
        let embedding = Embedding::new(node_index, &mut hope);
        if embedding.is_err() {
            log::error!("error : {:?}", embedding.as_ref().err());
            log::error!("hope embedding failed");
            std::process::exit(1);
        };
        let _embed_res = embedding.unwrap();
    }  // end case Hope
    else if sketching_params.is_some() {
        log::info!("embedding mode : Sketching");
        // now we embed
        let mut nodesketch = NodeSketch::new(sketching_params.unwrap(), trimat);
        let embedding = Embedding::new(node_index, &mut nodesketch);
        if embedding.is_err() {
            log::error!("error : {:?}", embedding.as_ref().err());
            log::error!("nodesketch embedding failed");
            std::process::exit(1);
        };
        let _embed_res = embedding.unwrap();
    }
    //    
}  // end fo main