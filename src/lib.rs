//! Spitz is a neural network library

#![crate_name = "spitz"]
#![crate_type = "lib"]
#![forbid(unsafe_code)]

use std::fmt;

// Ndarray
extern crate ndarray;
use ndarray::prelude::*;

// For random matrix
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

// Internal files
pub mod activations;

// * Layer struct
pub struct Layer {
    input: usize,
    size: usize,
    activation: &'static str,
}
impl Layer {
    pub fn new(input: usize, size: usize, activation: &'static str) -> Layer {
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

type Architecture = Vec<Layer>;
type Weights = Vec<Array2<f64>>;

/// Perceptron constitued of multiple layers
pub struct NNetwork {
    pub architecture: Architecture,
    pub weights: Weights,
}

impl Default for NNetwork {
    fn default() -> Self {
        Self {
            weights: Vec::new(),
            architecture: Vec::new(),
        }
    }
}

impl NNetwork {
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            architecture: Vec::new(),
        }
    }

    pub fn init_architecture(&mut self, arch_list: Vec<i32>) {
        let mut architecture: Architecture = Architecture::new();
        let mut input = arch_list[0] as usize;
        let mut arch_list = arch_list.iter();
        arch_list.next();
        for l in arch_list {
            let layer: Layer = Layer::new(input, *l as usize, "relu");
            architecture.push(layer);
            input = *l as usize;
        }
        self.architecture = architecture
    }

    pub fn init_weights(&mut self) {
        let mut weights: Weights = Vec::new();
        for layer in &self.architecture {
            let m = layer.input;
            let n = layer.size;

            let w: Array2<f64> = Array::random((m, n), Uniform::new(-1., 1.));

            weights.push(w);
        }
        self.weights = weights;
    }
}
