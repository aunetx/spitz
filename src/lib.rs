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

/// Perceptron constitued of multiple layers.
pub struct NNetwork {
    pub layer_structure: Vec<i32>,
    pub learning_rate: f64,
    pub epochs: usize,
    // TODO maybe make datas private (and provide call)
    pub datas: Datas,
    // Private; is used internally
    // TODO make them private
    pub is_test: bool,
    pub weights: Weights,
    pub architecture: Architecture,
    epoch: usize,
    grads: Weights,
}

impl Default for NNetwork {
    fn default() -> Self {
        Self {
            layer_structure: Vec::new(),
            weights: Vec::new(),
            datas: Datas::new(),
            learning_rate: 0.03,
            epochs: 15,
            is_test: false,
            architecture: Vec::new(),
            epoch: 0,
            grads: Vec::new(),
        }
    }
}

// Main functions
impl NNetwork {
    /// Returns a new uninitialized NNetwork object.
    pub fn new() -> Self {
        Self {
            layer_structure: Vec::new(),
            weights: Vec::new(),
            datas: Datas::new(),
            learning_rate: 0.03,
            epochs: 15,
            is_test: false,
            architecture: Architecture::new(),
            epoch: 0,
            grads: Vec::new(),
        }
    }

    /// ## Feed forward the network
    /// Runs the network with given `input`, and results `output`.
    pub fn feed_forward(&self, inputs: &Array2<f64>) -> Vec<Array2<f64>> {
        let mut x = vec![inputs.clone()];
        let mut z: Array2<f64>;
        let mut y: Array2<f64>;

        for w in &self.weights {
            // Weighted average `z = w · x`
            z = x.last().unwrap().dot(w);
            // Activation function `y = g(x)`
            y = activations::relu(z, false);
            // Append `y` to previous layers
            x.push(y.clone());
        }
        x
    }

    /// ## Calculate weights errors
    pub fn grads(&mut self) {
        // Forward propagation to get network datas
        let y: Weights = self.feed_forward(&self.datas.train_x);

        // Calculate global error
        let mut delta: Array2<f64> = y.last().unwrap().clone() - self.datas.train_y.view(); //.mapv(|a| a.powi(2));

        println!("Error = {}", delta);

        // Calculate error of output weights layer

        // TODO verify that we are not out of bounds
        self.grads[self.weights.len() - 1] = y[y.len() - 1].clone().t().dot(&delta);

        // Backpropagation of error
        // TODO make a for-loop for efficiency and readability
        let mut loop_lenght = self.grads.len() - 1;
        while loop_lenght > 0 {
            delta = delta.dot(&self.weights[loop_lenght].t())
                * activations::relu(y[loop_lenght].clone(), true);

            self.grads[loop_lenght - 1] = y[loop_lenght - 1].t().dot(&delta);
            loop_lenght -= 1;
        }

        // TODO Return grads ( grads / x.len() ?)
        /*for layer in &mut self.grads {
            layer.mapv(|x| x / self.datas.train_x.len() as f64);
        }*/
    }

    /// ## Train the network
    /// Trains the network over the previously given datasets
    pub fn fit(&mut self) {
        for epoch in 0..self.epochs {
            self.epoch = epoch;
            // Get errors for each layer
            self.grads();

            // Update weights for each layer
            #[allow(clippy::needless_range_loop)]
            for id in 0..self.weights.len() - 1 {
                assert_eq!(self.grads[id].shape(), self.weights[id].shape());
                self.weights[id] =
                    self.weights[id].clone() - self.grads[id].mapv(|x| x * self.learning_rate);
            }
        }
    }
}

// Public calls
impl NNetwork {
    // ## Init each part of the network
    pub fn init(&mut self) {
        // Init architecture
        self.init_architecture();
        // Init weights
        self.init_weights();
        // Init grads
        for _ in 0..self.weights.len() {
            self.grads.push(array![[]]);
        }
    }

    pub fn set_architecture(&mut self, arch: Vec<i32>) {
        self.layer_structure = arch;
    }

    /// Set learning rate.
    pub fn learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate
    }

    /// Set epochs number.
    pub fn epochs(&mut self, epochs: i32) {
        self.epochs = epochs as usize;
    }

    // TODO set dataset as `view only`
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
    pub fn import_datas(&mut self, x: Array2<f64>, y: Array2<f64>, test_ratio: Option<f64>) {
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
                self.datas.train_x = x;
                self.datas.train_y = y;
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
    }
}

// Private initializaters
impl NNetwork {
    /// Inits layers' inputs, size and activation function.
    fn init_architecture(&mut self) {
        let mut input = self.layer_structure[0] as usize;
        let mut arch_list = self.layer_structure.iter();
        arch_list.next();
        for l in arch_list {
            // Define multiple activations
            self.architecture
                .push(Layer::new(input, *l as usize, "relu"));
            input = *l as usize;
        }
    }

    /// Inits weights' matrices.
    fn init_weights(&mut self) {
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
}

// TODO create public callers for private elements of NNetwork
/*
// Public callers
impl NNetwork {
    fn get_architecture() {}
}
pub enum Get {}
impl Get {
    pub fn architecture(&self) {
        self
    }
}
*/
