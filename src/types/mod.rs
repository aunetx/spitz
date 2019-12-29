use crate::maths;
use ndarray::prelude::{array, Array2, Axis};

// * Layer struct
/// Structure describing a layer, contains : `input`, `size`, `activation`.\
/// Mostly used internally.
#[derive(Clone, Debug)]
pub struct Layer {
    pub input: usize,
    pub size: usize,
    pub activation: maths::TransfertFunction,
}
impl Layer {
    /// Returns a new `Layer` structure with given `input`, `size` and `activation`.
    pub fn new(input: usize, size: usize, activation: maths::Activation) -> Self {
        Self {
            input,
            size,
            activation: maths::Activation::match_activation(activation),
        }
    }
}

// * Architecture struct
#[derive(Debug, Clone)]
pub struct Architecture {
    pub layers: Vec<Layer>,
    input_layer_size: Option<usize>,
}
impl Default for Architecture {
    fn default() -> Self {
        Self {
            layers: Default::default(),
            input_layer_size: None,
        }
    }
}
impl Architecture {
    // TODO auto-detect input and output layer size
    pub fn add_layer(&mut self, neurons: usize, activation: maths::Activation) -> Result<(), &str> {
        let input: usize = match self.layers.last() {
            Some(l) => l.size,
            None => {
                match self.input_layer_size {
                    Some(n) => n,
                    None => return Err("cannot determine input layer size. Please set it with _network_.input_layer(size)")
                }
            },
        };
        self.layers.push(Layer::new(input, neurons, activation));
        Ok(())
    }

    pub fn input_layer(&mut self, neurons: usize) {
        self.input_layer_size = Some(neurons);
    }
}

// * Weights type
pub type Weights = Vec<Array2<f64>>;

// * DatasRaw struct
/// Structure describing training and test dataset.\
/// To set it, use `NNetwork.import_datas`.
#[derive(Debug, Clone)]
pub struct DatasRaw {
    pub train_x: Array2<f64>,
    pub train_y: Array2<f64>,
    pub test_x: Array2<f64>,
    pub test_y: Array2<f64>,
}
impl Default for DatasRaw {
    fn default() -> Self {
        Self {
            train_x: array![[]],
            train_y: array![[]],
            test_x: array![[]],
            test_y: array![[]],
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DatasTrain {
    pub x: Array2<f64>,
    pub y: Array2<f64>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DatasTest {
    pub x: Array2<f64>,
    pub y: Array2<f64>,
}

// * Datas struct
/// Structure similar to `DatasRaw` but that divides dataset for each epoch.\
/// Is used during training.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Datas {
    pub train: Vec<DatasTrain>,
    pub test: DatasTest,
}

impl Datas {
    /// Divides datas for each epoch.
    pub fn from_datas_raw(&mut self, datas_raw: &DatasRaw, epochs: usize) {
        self.test.x = datas_raw.test_x.clone();
        self.test.y = datas_raw.test_y.clone();

        // train_x
        let mut tr_x = Vec::new();
        for element in datas_raw.train_x.axis_chunks_iter(Axis(0), epochs) {
            tr_x.push(element.to_owned());
        }

        // train_y
        let mut tr_y = Vec::new();
        for element in datas_raw.train_y.axis_chunks_iter(Axis(0), epochs) {
            tr_y.push(element.to_owned());
        }

        let mut id = 0;
        while id < tr_x.len() {
            self.train.push(DatasTrain {
                x: tr_x[id].clone(),
                y: tr_y[id].clone(),
            });
            id += 1;
        }
    }
}
