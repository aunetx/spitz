pub mod activations;
use crate::Array2;

pub type TransfertFunction = fn(Array2<f64>, bool) -> Array2<f64>;

#[derive(Clone, Debug)]
/// List the different implemented transfert function to use.
pub enum Activation {
    /// ### Relu transfert function :
    /// For each `x` element, returns `x` if `x > 0`, else `0`.\
    /// One of the most used activations, works with pretty everything.
    ///
    /// #### Mathematically :
    /// `f(x) = max(x, 0)`
    /// #### Derivative :
    /// `∂f/∂x = 1 if x > 0, else 0`
    Relu,
    /// ### Sigmoid transfert function :
    /// For each `x` element, returns `1 / (1 + exp( -x ))`.\
    /// One of the most used activations, but does not fit every use case.
    ///
    /// #### Mathematically :
    /// `f(x) = 1 / (1 + exp( -x ))`
    /// #### Derivative :
    /// `∂f/∂x = f(x) * (1 - f(x))`
    Sigmoid,
    /// ### Linear transfert function :
    /// For each `x` element, returns `x`.\
    /// Barely used, excepted for testing or simple problems.
    ///
    /// #### Mathematically :
    /// `f(x) = x`
    /// #### Derivative :
    /// `∂f/∂x = 1`
    Linear,
}
impl Activation {
    pub fn match_activation(act_type: Activation) -> TransfertFunction {
        match act_type {
            Activation::Relu => activations::relu,
            Activation::Sigmoid => activations::sigmoid,
            Activation::Linear => activations::linear,
        }
    }
}
