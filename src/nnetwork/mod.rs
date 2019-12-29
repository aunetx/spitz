//! ### NNetwork
//! Provides most parts of `NNetwork` struct, it is the main class of the library.

use crate::{log::*, Array2, NNetwork, Weights};

impl Default for NNetwork {
    fn default() -> Self {
        Self {
            architecture: Default::default(),
            weights: Default::default(),
            datas_raw: Default::default(),
            datas: Default::default(),
            grads: Default::default(),
            learning_rate: crate::DEFAULT_LN,
            epochs: crate::DEFAULT_EPOCHS,
            batches: crate::DEFAULT_BATCHES,
            epoch: 0,
            batch: 0,
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
        // TODO create `x` during init so we don't need to create again it during each batch of each epoch
        let mut x = vec![inputs.clone()];
        let mut z: Array2<f64>;
        let mut y: Array2<f64>;

        for (id, w) in self.weights.iter().enumerate() {
            // Weighted average `z = w · x`
            z = x.last().unwrap().dot(w);
            // Activation function `y = g(x)`
            y = (self.architecture.layers[id].activation)(z, false);
            // Append `y` to previous layers
            x.push(y);
        }
        x
    }

    /// Calculate weights errors
    pub fn grads(&mut self) {
        let data = &self.datas.train[self.batch];

        // Forward propagation to get network datas
        let y: Weights = self.feed_forward(&data.x);

        // Calculate global error
        let mut delta: Array2<f64> = (y.last().unwrap() - &data.y).mapv(|a| a.powi(2));

        // Calculate error of output weights layer
        self.grads[self.weights.len() - 1] = y[y.len() - 1].t().dot(&delta);

        // Backpropagation of error
        // TODO make a for-loop for efficiency and readability
        let mut loop_lenght = self.grads.len() - 1;
        while loop_lenght > 0 {
            // FIXME verify that the activation used is the good one and the same as the derivative used
            delta = delta.dot(&self.weights[loop_lenght].t())
                * (self.architecture.layers[loop_lenght].activation)(y[loop_lenght].clone(), true);

            self.grads[loop_lenght - 1] = y[loop_lenght - 1].t().dot(&delta);
            loop_lenght -= 1;
        }

        let len = data.x.len() as f64;
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

            debug!("epoch n°{}", self.epoch);

            for batch in 0..self.datas.train.len() {
                self.batch = batch;

                trace!("batch n°{}", self.batch);

                // Get errors for each layer
                self.grads();

                // Update weights for each layer
                for id in 0..self.weights.len() - 1 {
                    self.weights[id] =
                        &self.weights[id] - &self.grads[id].mapv(|x| x * self.learning_rate);
                }
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
