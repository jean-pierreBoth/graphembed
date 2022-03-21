//  test for datafiles


use clap::{Arg,Command};

use graphite::prelude::*;
use crate::{nodesketch::*};

static DATADIR : &str = &"/home/jpboth/Rust/graphembed/Data";


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
            .short('e')
            .required(true)
            .takes_value(true)
            .help("specify \"hope\" or \"sketching\" "))
        .arg(Arg::new("sketching")
            .long("sketch")
            .short('s')
            .takes_value(true)
            .help("expecting dimension of sketching"))
        .subcommand_value_name("hope")
            .arg(Arg::new("mode")
                .long("mode")
                .takes_value(true)
                .help("expecting hope arguments"))
        .subcommand_value_name("sketching")
            .arg(Arg::new("dim")
            .takes_value(true)
            .required(true)
            .help("expectiong embedding dimension"))

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
    }

    let mode;
    if matches.is_present("embedder") {
        let mode_str = matches.value_of("embedder").ok_or("").unwrap().parse::<String>().unwrap();
        match mode_str.as_str() {
            "hope" =>  {
                println!("got hope mode");
                mode = EmbeddingMode::Hope;
            }
            "sketching" => {
                println!("got sketching mode");
                std::process::exit(1);
            }
            _           => {
                println!("got sketching mode neither hope nor sketching");
                std::process::exit(1);
            } 
        } // end match 
    } 
    else {
        println!("embedding argument necessary");
        std::process::exit(1);
    }


    let mut sketch_dimension : usize = 0;
    if matches.is_present("sketching") {
        sketch_dimension = match matches.value_of("").ok_or("").unwrap().parse::<usize>() {
                Ok(dim)  => { dim},
                    _          => { 
                                    println!("could not decode embedding dimension, exiting");
                                    std::process::exit(1);
                                }
        } // end of match
    } // end decode sketching

    

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
            let sketch_size = sketch_dimension;
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