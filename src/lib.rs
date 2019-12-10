//! Spitz is a neural network library

#![crate_name = "spitz"]
#![crate_type = "lib"]
#![forbid(unsafe_code)]

// ndarray
use ndarray::prelude::*;

// ndarray_rand
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

// Internal files
mod interfaces;
pub mod maths;
pub mod nnetwork;
pub mod types;

pub use interfaces::{PrivateCalls, PublicCalls};
use types::*;

/// Perceptron constitued of multiple layers.
pub struct NNetwork {
    pub layer_structure: Vec<i32>,
    pub learning_rate: f64,
    pub epochs: usize,
    // TODO maybe make datas private (and provide call)
    pub datas: Datas,
    // Private; is used internally
    // TODO make them private
    is_test: bool,
    weights: Weights,
    architecture: Architecture,
    epoch: usize,
    grads: Weights,
}
