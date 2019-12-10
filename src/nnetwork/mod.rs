//! ### NNetwork
//! Provides most parts of `NNetwork` struct, it is the main class of the library.

use crate::{maths, Array2, NNetwork, Weights};
use maths::*;

impl Default for NNetwork {
    fn default() -> Self {
        Self {
            layer_structure: Default::default(),
            architecture: Default::default(),
            weights: Default::default(),
            datas: Default::default(),
            grads: Default::default(),
            learning_rate: 0.03,
            is_test: false,
            epochs: 15,
            epoch: 0,
        }
    }
}

impl NNetwork {
    /// Returns a new uninitialized NNetwork object.
    pub fn new() -> Self {
        Default::default()
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
        let len = self.datas.train_x.len();
        for layer in &mut self.grads {
            layer.mapv(|x| x / len as f64);
        }
    }

    /// ## Train the network
    /// Trains the network over the previously given datasets
    pub fn fit(&mut self) -> &mut Self {
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
        self
    }
}
