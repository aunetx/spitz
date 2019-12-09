use ndarray::prelude::Array2;

/// Utilitaries for activation functions.
mod utils {
    /// Returns `x` if `x > 0`, else `0`.
    pub fn max(x: f64) -> f64 {
        if x < 0. {
            0.
        } else {
            x
        }
    }

    /// Returns `1` if `x > 0`, else `0`.
    pub fn sup(x: f64) -> f64 {
        if x < 0. {
            0.
        } else {
            1.
        }
    }
}

/// ### Relu transfert function :
/// For each `x` element, returns `x` if `x > 0`, else `0`.\
/// One of the most used activations, works with pretty everything.
///
/// #### Mathematically :
/// `f(x) = max(x, 0)`
/// #### Derivative :
/// `∂f/∂x = 1 if x > 0, else 0`
pub fn relu(x: Array2<f64>, derivative: bool) -> Array2<f64> {
    if !derivative {
        x.mapv(utils::max)
    } else {
        x.mapv(utils::sup)
    }
}

pub fn sigmoid(x: Array2<f64>, derivative: bool) -> Array2<f64> {
    let sig = |a: f64| 1. / (1. + (-a).exp());
    if !derivative {
        x.mapv(|a| sig(a) * (1. - sig(a)))
    } else {
        x.mapv(sig)
    }
}
