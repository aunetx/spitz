//! Spitz is a neural network library

#![crate_name = "spitz"]
#![crate_type = "lib"]
#![forbid(unsafe_code)]

const WEIGHTS_INIT_MULTIPLIER: f64 = 0.1;
const DEFAULT_LN: f64 = 0.03;
const DEFAULT_EPOCHS: usize = 15;

// ndarray
use ndarray::prelude::*;

// ndarray_rand
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

// log
extern crate log;

// Internal files
mod interfaces;
pub mod maths;
pub mod nnetwork;
mod types;
pub use interfaces::{PrivateCalls, PublicCalls};
pub use maths::Activation;
pub use types::*;

/// Perceptron constitued of multiple layers.
#[derive(Clone)]
pub struct NNetwork {
    pub learning_rate: f64,
    pub epochs: usize,
    pub datas: Datas,
    // Private; is used internally
    is_test: bool,
    /// Call with
    weights: Weights,
    architecture: Architecture,
    epoch: usize,
    grads: Weights,
}
