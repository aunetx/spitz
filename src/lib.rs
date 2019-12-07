//! Spitz is a neural network library

#![crate_name = "spitz"]
#![crate_type = "lib"]
#![forbid(unsafe_code)]

extern crate ndarray;
use ndarray::prelude::*;

// For random matrix
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

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

type Architecture = Vec<Layer>;
type Weights = Vec<Array2<f64>>;

/// Perceptron constitued of multiple layers
pub struct NNetwork {
    architecture: Architecture,
    pub weights: Weights,
}

impl NNetwork {
    pub fn new(arch: Architecture) -> NNetwork {
        Self {
            weights: Self::init_weights(&arch),
            architecture: arch,
        }
    }

    fn init_weights(arch: &[Layer]) -> Weights {
        let mut weights: Weights = Vec::new();
        for layer in arch {
            let m = layer.input;
            let n = layer.size;

            let w: Array2<f64> = Array::random((m, n), Uniform::new(-1., 1.));

            weights.push(w);
        }
        weights
    }
}
