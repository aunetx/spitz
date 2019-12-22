//! ### NNetwork
//! Provides most parts of `NNetwork` struct, it is the main class of the library.

use crate::{log::*, Array2, NNetwork, Weights};

impl Default for NNetwork {
    fn default() -> Self {
        Self {
            architecture: Default::default(),
            weights: Default::default(),
            datas: Default::default(),
            grads: Default::default(),
            learning_rate: crate::DEFAULT_LN,
            epochs: crate::DEFAULT_EPOCHS,
            is_test: false,
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

        for (id, w) in self.weights.iter().enumerate() {
            // Weighted average `z = w · x`
            z = x.last().unwrap().dot(w);
            // Activation function `y = g(x)`
            y = (self.architecture.layers[id].activation)(z, true);
            // Append `y` to previous layers
            x.push(y.clone());
        }
        x
    }

    /// ## Calculate weights errors
    // FIXME change maths operations : does not work well
    pub fn grads(&mut self) {
        // Forward propagation to get network datas
        let y: Weights = self.feed_forward(&self.datas.train_x);

        // Calculate global error
        let mut delta: Array2<f64> = y.last().unwrap().clone() - self.datas.train_y.view(); //.mapv(|a| a.powi(2));

        // ! info!("Error for epoch {} :\n{}\n", self.epoch, delta);

        // Calculate error of output weights layer

        self.grads[self.weights.len() - 1] = y[y.len() - 1].clone().t().dot(&delta);

        // Backpropagation of error
        // TODO make a for-loop for efficiency and readability
        let mut loop_lenght = self.grads.len() - 1;
        while loop_lenght > 0 {
            delta = delta.dot(&self.weights[loop_lenght].t())
                * (self.architecture.layers[loop_lenght].activation)(y[loop_lenght].clone(), true);
            // FIXME verify that the activation used is the same as the derivative used
            // ! Verify also that it is the one that the user defined
            debug!(
                "Layer activation : {:?}\nUsed activation : {:?}",
                loop_lenght, self.architecture.layers[loop_lenght].activation
            );

            self.grads[loop_lenght - 1] = y[loop_lenght - 1].t().dot(&delta);
            loop_lenght -= 1;
        }

        let len = self.datas.train_x.len() as f64;
        for layer in &mut self.grads {
            layer.mapv(|x| x / len);
        }
    }

    /// ## Train the network
    /// Trains the network over the previously given datasets
    // TODO make training a concurrent process
    pub fn fit(&mut self) -> &mut Self {
        for epoch in 0..self.epochs {
            self.epoch = epoch;
            // Get errors for each layer
            self.grads();

            // Update weights for each layer
            for id in 0..self.weights.len() - 1 {
                assert_eq!(self.grads[id].shape(), self.weights[id].shape());
                self.weights[id] =
                    self.weights[id].clone() - self.grads[id].mapv(|x| x * self.learning_rate);
            }
        }
        self
    }

    /// Print weights (used mostly for debugging).
    pub fn print_weights(&mut self) -> &mut Self {
        for (id, w) in self.weights.iter().enumerate() {
            println!("Layer {} to {}\n{:7.4}\n", id, id + 1, w);
        }
        println!("\n");
        self
    }
}
