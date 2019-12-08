//! Spitz is a neural network library

#![crate_name = "spitz"]
#![crate_type = "lib"]
#![forbid(unsafe_code)]

// Ndarray
extern crate ndarray;
use ndarray::prelude::*;

// To init random matrices
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

// Internal files
pub mod activations;
pub mod architecture;
use architecture::*;

/// Perceptron constitued of multiple layers
pub struct NNetwork {
    pub architecture: Architecture,
    pub weights: Weights,
    pub datas: Datas,
}

impl Default for NNetwork {
    fn default() -> Self {
        Self {
            weights: Vec::new(),
            architecture: Vec::new(),
            datas: Datas::new(),
        }
    }
}

impl NNetwork {
    /// Returns a new uninitialized NNetwork object
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            architecture: Vec::new(),
            datas: Datas::new(),
        }
    }

    /// Inits layers' inputs, size and activation function
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

    /// Inits weights' matrices
    pub fn init_weights(&mut self) {
        let mut weights: Weights = Vec::new();
        for layer in &self.architecture {
            let m = layer.input;
            let n = layer.size;

            // TODO add `multiplier` argument (or nnetwork variable) to initialize weights' values distribution
            let w: Array2<f64> = Array::random((m, n), Uniform::new(-1., 1.)) * 0.1;

            weights.push(w);
        }
        self.weights = weights;
    }

    /// ## Imports data from `x` and `y` 2D arrays.
    /// `test_ratio` the ratio data extracted that is used to test the network, what's left is used to train the network.\
    /// Usually, `test_ratio` is :
    /// - around than `0.1` for medium-sized dataset, `< 100_000`
    /// - less than `0.05` for big-sized dataset, `> 100_000`
    ///
    /// The minimum number of datas selected for testing will be `1`.
    /// The minimum number of datas selected for training will be `1`.
    ///
    /// ### Returns
    /// Returns a tuple containing the number of testing and training selected datas.
    ///
    /// ### Panics
    /// Panics if `test_ratio` is not bewteen `0` and `1`.\
    /// Panics if `x` and `y` are not the same shape.
    pub fn import_data(&mut self, x: Array2<f64>, y: Array2<f64>, test_ratio: f64) -> (i32, i32) {
        // Panics test
        if (test_ratio <= 0.) || (test_ratio >= 1.) {
            panic!(
                "test ratio must be between 0.0 and 1.0 (ratio = {})",
                test_ratio
            );
        }
        if x.shape()[0] != y.shape()[0] {
            panic!(
                "x and y must be aligned ({} != {})",
                x.shape()[0],
                y.shape()[0]
            )
        }

        // Get number of datas to extract
        let mut test_number = (x.shape()[0] as f64 * test_ratio).round() as i32;
        if test_number == 0 {
            test_number = 1;
        } else if test_number == x.shape()[0] as i32 {
            test_number -= 1;
        };
        let mut train_number =
            (x.shape()[0] as f64 - x.shape()[0] as f64 * test_ratio).round() as i32;
        if train_number == x.shape()[0] as i32 {
            train_number -= 1;
        } else if train_number == 0 {
            train_number = 1;
        }

        // Extract and set datas
        self.datas.test_x = x
            .slice_axis(Axis(0), ndarray::Slice::from(-test_number..))
            .to_owned();
        self.datas.test_y = y
            .slice_axis(Axis(0), ndarray::Slice::from(-test_number..))
            .to_owned();
        self.datas.train_x = x
            .slice_axis(Axis(0), ndarray::Slice::from(0..train_number))
            .to_owned();
        self.datas.train_y = y
            .slice_axis(Axis(0), ndarray::Slice::from(0..train_number))
            .to_owned();
        (train_number, test_number)
    }
}
