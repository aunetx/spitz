//! ### Interfaces
//! Provides some functions used to interface with the user and the library.\
//! Divided in two groups :
//! - `PublicCalls`, to interface with the user : import datas, set parameters ;
//! - `PrivateCalls`, provides private functions to init the network.

use crate::types::*;
use crate::{array, Array, Array2, Axis, RandomExt, Uniform};

/// Public callers (get and set methods).
pub trait PublicCalls {
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
    /// Panics if `test_ratio` is not between `0` and `1`.\
    /// Panics if `x` and `y` are not the same shape.
    fn import_datas(
        &mut self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        test_ratio: Option<f64>,
    ) -> &mut Self;
    /// Gives an architecture to the network from an array of `i32`.
    fn set_architecture(&mut self, arch: Vec<i32>) -> &mut Self;
    /// Set learning rate.
    fn set_learning_rate(&mut self, rate: f64) -> &mut Self;
    /// Set epochs number.
    fn set_epochs(&mut self, epochs: i32) -> &mut Self;
    /// Add a layer to the architecture
    fn add_layer(&mut self, neurons: usize, activation: &'static str) -> &mut Self;
    /// Define input layer size of the architecture
    fn input_layer(&mut self, neurons: usize) -> &mut Self;

    // ## Init each part of the network
    fn init(&mut self) -> &mut Self;

    /// Returns architecture of given network.
    fn get_architecture(&self) -> Architecture;
    /// Returns weights of given network.
    fn get_weights(&self) -> Weights;
    /// Returns whether test is used or not.
    fn get_is_test(&self) -> bool;
}

impl PublicCalls for crate::NNetwork {
    // TODO set dataset as `view only`
    fn import_datas(
        &mut self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        test_ratio: Option<f64>,
    ) -> &mut Self {
        // Panics test
        match test_ratio {
            None => self.is_test = false,
            Some(test_ratio) => {
                if (test_ratio <= 0.) || (test_ratio >= 1.) {
                    panic!(
                        "test ratio must be between 0.0 and 1.0 (ratio = {})",
                        test_ratio
                    );
                };
                self.is_test = true
            }
        };
        if x.shape()[0] != y.shape()[0] {
            panic!(
                "x and y must be aligned ({} != {})",
                x.shape()[0],
                y.shape()[0]
            )
        };

        // Extract datas and set them
        match test_ratio {
            // In case we don't want test dataset
            None => {
                self.datas.test_x = array![[]];
                self.datas.test_y = array![[]];
                self.datas.train_x = x.to_owned();
                self.datas.train_y = y.to_owned();
            }
            // In case we want a test dataset given by ratio
            Some(test_ratio) => {
                // Get the number of test datas
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
            }
        };
        self
    }
    fn set_architecture(&mut self, arch: Vec<i32>) -> &mut Self {
        self.layer_structure = arch;
        self
    }
    fn set_learning_rate(&mut self, rate: f64) -> &mut Self {
        self.learning_rate = rate;
        self
    }
    fn set_epochs(&mut self, epochs: i32) -> &mut Self {
        self.epochs = epochs as usize;
        self
    }
    fn add_layer(&mut self, neurons: usize, activation: &'static str) -> &mut Self {
        match self.architecture.add_layer(neurons, activation) {
            Ok(()) => self,
            Err(e) => panic!("`add_layer` : {}", e),
        }
    }
    fn input_layer(&mut self, neurons: usize) -> &mut Self {
        self.architecture.input_layer(neurons);
        self
    }
    fn init(&mut self) -> &mut Self {
        self.init_weights();
        // Init grads array
        for _ in 0..self.weights.len() {
            self.grads.push(array![[]]);
        }
        self
    }

    fn get_architecture(&self) -> Architecture {
        self.architecture.clone()
    }
    fn get_weights(&self) -> Weights {
        self.weights.clone()
    }
    fn get_is_test(&self) -> bool {
        self.is_test
    }
}

/// Private callers (initializers).
pub trait PrivateCalls {
    /// Inits weights' matrices.
    fn init_weights(&mut self);
}

impl PrivateCalls for crate::NNetwork {
    /// Inits weights' matrices.
    fn init_weights(&mut self) {
        for layer in &self.architecture.layers {
            let m = layer.input;
            let n = layer.size;

            let w: Array2<f64> =
                Array::random((m, n), Uniform::new(-1., 1.)) * crate::WEIGHTS_INIT_MULTIPLIER;

            self.weights.push(w);
        }
    }
}
