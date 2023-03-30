//! To describe dump of embedding 


/// only Bson now.
#[derive(Copy, Clone)]
pub enum Format {
    BSON,
}

pub struct Output {
    /// describe output format 
    fmt : Format,
    /// do we dump indexation?
    indexation : bool, 
    /// name of output file
    output_name : String,
}

impl Output {
    /// if output_name is None, default output_name will be "embedding.bson"
    pub fn new(fmt : Format, indexation : bool, output_name : &Option<String>) -> Self {
        let output_name = match output_name  {
            Some(name) => {
                    let mut bson_name = name.clone();
                    bson_name.push_str(".bson");
                    bson_name
            },
            None => String::from("embedding.bson"),
        };
        Output{fmt, indexation, output_name : output_name} 
    }
    /// get ouput format
    pub fn get_fmt(&self) -> Format { self.fmt}

    /// get output_name
    pub fn get_output_name(&self) -> &String { &self.output_name}

    /// get indexation 
    pub fn get_indexation(&self) -> bool { self.indexation}

}  // end of Output


impl Default for Output {
    fn default() -> Self {
        Output{fmt : Format::BSON, indexation: true, output_name : String::from("embedding.bson")}
    }
}