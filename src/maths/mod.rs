pub mod activations;
use crate::Array2;

pub type TransfertFunction = fn(Array2<f64>, bool) -> Array2<f64>;

#[derive(Clone, Debug)]
pub enum Activation {
    Relu,
    Sigmoid,
}
impl Activation {
    pub fn match_activation(act_type: Activation) -> TransfertFunction {
        match act_type {
            Activation::Relu => activations::relu,
            Activation::Sigmoid => activations::sigmoid,
        }
    }
}
