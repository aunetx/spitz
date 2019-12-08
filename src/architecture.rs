use ndarray::prelude::*;
use std::fmt;

// * Layer struct
/// Structure describing a layer, contains : `input`, `size`, `activation`.\
/// Mostly used internally.
pub struct Layer {
    pub input: usize,
    pub size: usize,
    pub activation: &'static str,
}
impl Layer {
    /// Returns a new `Layer` structure with given `input`, `size` and `activation`.
    pub fn new(input: usize, size: usize, activation: &'static str) -> Self {
        Self {
            input,
            size,
            activation,
        }
    }
}
impl fmt::Debug for Layer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[{} neurons, {} input, {:?}]",
            self.size, self.input, self.activation
        )
    }
}

pub type Architecture = Vec<Layer>;
pub type Weights = Vec<Array2<f64>>;

// * Datas struct
/// Structure describing training and test dataset.\
/// To set it, use `NNetwork.import_datas`.
pub struct Datas {
    pub train_x: Array2<f64>,
    pub train_y: Array2<f64>,
    pub test_x: Array2<f64>,
    pub test_y: Array2<f64>,
}
impl Default for Datas {
    fn default() -> Self {
        Self {
            train_x: array![[]],
            train_y: array![[]],
            test_x: array![[]],
            test_y: array![[]],
        }
    }
}
impl Datas {
    /// Returns a new empty `Datas` structure.
    pub fn new() -> Self {
        Self {
            train_x: array![[]],
            train_y: array![[]],
            test_x: array![[]],
            test_y: array![[]],
        }
    }
}
