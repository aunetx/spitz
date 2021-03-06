//! ### Interfaces
//! Provides some functions used to interface with the user and the library.\
//! Divided in two groups :
//! - `PublicCalls`, to interface with the user : import datas, set parameters ;
//! - `PrivateCalls`, provides private functions to init the network.

use crate::types::*;
use crate::{array, log::*, maths, Array, Array2, Axis, RandomExt, Uniform};

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
    fn import_datas(&mut self, x: &Array2<f64>, y: &Array2<f64>, test_ratio: f64) -> &mut Self;
    // TODO documentation for `import_train_datas` and `import_test_datas`
    /// Import only training datas
    fn import_train_datas(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> &mut Self;
    /// Import only testing datas
    fn import_test_datas(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> &mut Self;
    /// Set learning rate.
    fn set_learning_rate(&mut self, rate: f64) -> &mut Self;
    /// Set epochs number.
    fn set_epochs(&mut self, epochs: i32) -> &mut Self;
    /// Set the number of datas per batch.
    fn set_batches(&mut self, batches: i32) -> &mut Self;
    /// Add a layer to the architecture.
    fn add_layer(&mut self, neurons: usize, activation: maths::Activation) -> &mut Self;
    /// Define input layer size of the architecture.
    fn input_layer(&mut self, neurons: usize) -> &mut Self;

    /// Init each part of the network.
    fn init(&mut self) -> &mut Self;

    /// Returns architecture of given network.
    fn get_architecture(&self) -> Architecture;
    /// Returns weights of given network.
    fn get_weights(&self) -> Weights;
}

impl PublicCalls for crate::NNetwork {
    fn import_datas(&mut self, x: &Array2<f64>, y: &Array2<f64>, test_ratio: f64) -> &mut Self {
        // Panics test
        if (test_ratio <= 0.) || (test_ratio >= 1.) {
            panic!(
                "test ratio must be between 0.0 and 1.0 (ratio = {})",
                test_ratio
            );
        };
        if x.shape()[0] != y.shape()[0] {
            panic!(
                "x and y must be aligned ({} != {})",
                x.shape()[0],
                y.shape()[0]
            )
        };
        trace!("Test ratio = {:?}", test_ratio);

        // Extract datas and set them

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
        self.datas_raw.test_x = x
            .slice_axis(Axis(0), ndarray::Slice::from(-test_number..))
            .to_owned();
        self.datas_raw.test_y = y
            .slice_axis(Axis(0), ndarray::Slice::from(-test_number..))
            .to_owned();
        self.datas_raw.train_x = x
            .slice_axis(Axis(0), ndarray::Slice::from(0..train_number))
            .to_owned();
        self.datas_raw.train_y = y
            .slice_axis(Axis(0), ndarray::Slice::from(0..train_number))
            .to_owned();

        self
    }
    fn import_train_datas(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> &mut Self {
        // Panics test
        if x.shape()[0] != y.shape()[0] {
            panic!(
                "x and y must be aligned ({} != {})",
                x.shape()[0],
                y.shape()[0]
            )
        };

        // Extract datas and set them
        self.datas_raw.train_x = x.to_owned();
        self.datas_raw.train_y = y.to_owned();

        self
    }
    fn import_test_datas(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> &mut Self {
        // Panics test
        if x.shape()[0] != y.shape()[0] {
            panic!(
                "x and y must be aligned ({} != {})",
                x.shape()[0],
                y.shape()[0]
            )
        };

        // Extract datas and set them
        self.datas_raw.test_x = x.to_owned();
        self.datas_raw.test_y = y.to_owned();

        self
    }
    fn set_learning_rate(&mut self, rate: f64) -> &mut Self {
        self.learning_rate = rate;
        trace!("Learning rate set : {:?}", rate);
        self
    }
    fn set_epochs(&mut self, epochs: i32) -> &mut Self {
        self.epochs = epochs as usize;
        trace!("Epochs number set : {:?}", epochs);
        self
    }
    fn set_batches(&mut self, batches: i32) -> &mut Self {
        self.batches = batches as usize;
        trace!("Batches number set : {:?}", batches);
        self
    }
    fn add_layer(&mut self, neurons: usize, activation: maths::Activation) -> &mut Self {
        match self.architecture.add_layer(neurons, activation.clone()) {
            Ok(()) => {
                trace!(
                    "Adding layer with {:?} neurons and activation {:?}",
                    neurons,
                    activation
                );
                self
            }
            Err(e) => panic!("`add_layer` : {}", e),
        }
    }
    fn input_layer(&mut self, neurons: usize) -> &mut Self {
        self.architecture.input_layer(neurons);
        trace!("Input layer set with {:?} neurons", neurons);
        self
    }
    fn init(&mut self) -> &mut Self {
        // Init `datas` from `datas_raw`
        self.datas.from_datas_raw(&self.datas_raw, self.batches);
        // Init weights
        self.init_weights();
        // Init grads array
        for _ in 0..self.weights.len() {
            self.grads.push(array![[]]);
        }
        trace!("Initiated network");
        self
    }

    fn get_architecture(&self) -> Architecture {
        self.architecture.clone()
    }
    fn get_weights(&self) -> Weights {
        self.weights.clone()
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

            let w: Array2<f64> = Array::random(
                (m, n),
                // TODO maybe change distribution to use
                Uniform::new(crate::WEIGHTS_INIT_MIN, crate::WEIGHTS_INIT_MAX),
            );

            self.weights.push(w);
        }
    }
}
