//  test for datafiles


use clap::{Arg, ArgMatches, Command, arg};

use graphite::prelude::*;
use crate::{nodesketch::*};

static DATADIR : &str = &"/home/jpboth/Rust/graphembed/Data";


fn parse_sketching(matches : &ArgMatches) -> Option<NodeSketchParams> {
    log::debug!("in parse_sketching");
    // get embedding dimension
    let dimension = match matches.value_of("dim") {
        Some(str) => {
            str.parse::<usize>().unwrap()
        },
        _   => { return None; },
    }; // end match
    // get decay
    let decay = match matches.value_of("decay") {
        Some(str) => {
            str.parse::<f64>().unwrap()
        },
        _   => { return None; },
    }; // end match 
    // get nbiter
    let nb_iter = match matches.value_of("nbiter") {
        Some(str) => {
            str.parse::<usize>().unwrap()
        },
        _   => { return None; },
    }; // end match
    //
    let sketch_params = NodeSketchParams{sketch_size: dimension, decay, nb_iter, parallel : true};
    return Some(sketch_params);
} // end of parse_sketching


fn parse_hope_args(matches : &ArgMatches)  -> Option<HopeParams> {
    log::debug!("in parse_hope");
    // first get mode Katz or Rooted Page Rank
    let mut epsil : f64 = 0.;
    let mut maxrank : usize = 0;
    let mut blockiter = 0;
    //
    match matches.subcommand() {
        Some(("precision", sub_m)) =>  {
            if let Some(str) = sub_m.value_of("epsil") {
                if str.parse::<usize>().is_ok() {
                    epsil = str.parse::<f64>().unwrap();
                }
                else {
                    return None;
                }            
            } // end of epsil
 
            // get maxrank
            if let Some(str) = sub_m.value_of("maxrank") {
                if str.parse::<usize>().is_ok() {
                    maxrank = str.parse::<usize>().unwrap();
                }
                else {
                    return None;
                }
            }

            // get blockiter
            if let Some(str) = sub_m.value_of("blockiter") {
                if str.parse::<usize>().is_ok() {
                   blockiter  = str.parse::<usize>().unwrap();
                }
                else {
                    return None;
                }         
            }
            let range = RangeApproxMode::EPSIL(RangePrecision::new(epsil, blockiter, maxrank));
            return None;
        },  // end decoding preciison arg


        Some(("rank", sub_m)) => {
            if let Some(str) = sub_m.value_of("maxrank") {
                if str.parse::<usize>().is_ok() {
                    maxrank = str.parse::<usize>().unwrap();
                }
                else {
                    return None;
                }
            }

            // get blockiter
            if let Some(str) = sub_m.value_of("blockiter") {
                if str.parse::<usize>().is_ok() {
                   blockiter  = str.parse::<usize>().unwrap();
                }
                else {
                    return None;
                }         
            }             
            let range = RangeApproxMode::RANK(RangeRank::new(maxrank, blockiter));
        }, // end of decoding rank arg

        _  => {
            log::error!("could not decode hope argument, got neither precision nor rank subcommands");
            return None;
        },

    }; // end match
    return None;
} // end of parse_hope_args



pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    log::info!("logger initialized"); 
    //
    let matches = Command::new("embed")
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
            .subcommand(Command::new("precision")
                .args(&[
                    arg!(--maxrank <maxrank> "maximum rank expected"),
                    arg!(--blockiter <blockiter> "integer between 2 and 5"),
                    arg!(-e --epsil <precision> "precision between 0. and 1."),
                ]))
            .subcommand(Command::new("rank")
                .args(&[
                    arg!(--targetrank <targetrank> "expected rank"),
                    arg!(--nbiter <nbiter> "integer between 2 and 5"),
                ]))
                
        )
        .subcommand(Command::new("sketching")
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

    let mode;
    match matches.subcommand() {
        Some(("hope", sub_m)) => {
            log::debug!("got hope mode");
            mode = EmbeddingMode::Hope;
            parse_hope_args(sub_m);
        },

        Some(("sketching", sub_m )) => {
            log::debug!("got sketching mode");
            mode = EmbeddingMode::Nodesketch;
            parse_sketching(sub_m);
        }
        _  => {
            log::error!("expected subcommand hope or nodesketch");
            std::process::exit(1);
        }
    }  // end match subcommand




    log::info!("in hope::test_hope_gnutella09"); 
    // Nodes: 8114 Edges: 26013
    let path = std::path::Path::new(crate::DATADIR).join(fname.clone().as_str());
    log::info!("\n\n test_nodesketchasym_wiki, loading file {:?}", path);
    let res = csv_to_trimat::<f64>(&path, true, b'\t');
    if res.is_err() {
        log::error!("error : {:?}", res.as_ref().err());
        log::error!("hope::tests::test_hope_gnutella09 failed in csv_to_trimat");
        assert_eq!(1, 0);
    }
    let (trimat, node_index) = res.unwrap();
    //
    // we have our graph in trimat format
    //
    match mode {
        EmbeddingMode::Hope => { 
            log::info!("embedding mode : Hope");
            // 
            let hope_m = HopeMode::KATZ;
            let decay_f = 1.;
            let range_m = RangeApproxMode::EPSIL(RangePrecision::new(0.5, 3, 4000));
            let params = HopeParams::new(hope_m, range_m, decay_f);
            // now we embed
            let mut hope = Hope::new(params, trimat); 
            let hope_embedding = Embedding::new(node_index, &mut hope);
            if hope_embedding.is_err() {
                log::error!("error : {:?}", hope_embedding.as_ref().err());
                log::error!("test_wiki failed in compute_Embedded");
                assert_eq!(1, 0);        
            };
            let _embed_res = hope_embedding.unwrap();
        },  // end case Hope

        EmbeddingMode::Nodesketch => {
            log::info!("embedding mode : Sketching");
            let sketch_size = 100;    // TODO
            let decay = 0.1;
            let nb_iter = 10;
            let parallel = false;
            let params = NodeSketchParams{sketch_size, decay, nb_iter, parallel};
            // now we embed
            let mut nodesketch = NodeSketch::new(params, trimat);
            let _sketch_embedding = Embedding::new(node_index, &mut nodesketch);
        }
    }; // end of match
    //    
}  // end fo main