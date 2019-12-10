use crate::types::*;
use crate::{array, Array, Array2, Axis, RandomExt, Uniform};

/// Public callers (get and set methods)
pub trait PublicCalls {
    /// Gives an architecture to the network from an array of `i32`.
    fn set_architecture(&mut self, arch: Vec<i32>);
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
    fn import_datas(&mut self, x: Array2<f64>, y: Array2<f64>, test_ratio: Option<f64>);

    /// Returns architecture of given network.
    fn get_architecture(&self) -> Architecture;
    /// Returns weights of given network.
    fn get_weights(&self) -> Weights;
    /// Returns whether test is used or not.
    fn get_is_test(&self) -> bool;
}

impl PublicCalls for crate::NNetwork {
    fn set_architecture(&mut self, arch: Vec<i32>) {
        self.layer_structure = arch;
    }
    // TODO set dataset as `view only`
    fn import_datas(&mut self, x: Array2<f64>, y: Array2<f64>, test_ratio: Option<f64>) {
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

/// Private callers (initializers)
pub trait PrivateCalls {
    /// Inits layers' inputs, size and activation function.
    fn init_architecture(&mut self);

    /// Inits weights' matrices.
    fn init_weights(&mut self);
}

impl PrivateCalls for crate::NNetwork {
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
        for layer in &self.architecture {
            let m = layer.input;
            let n = layer.size;

            // TODO add `multiplier` argument (or nnetwork variable) to initialize weights' values distribution
            let w: Array2<f64> = Array::random((m, n), Uniform::new(-1., 1.)) * 0.1;

            self.weights.push(w);
        }
    }
}
