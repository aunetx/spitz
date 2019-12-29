use ndarray::prelude::Array2;

/// Utilitaries for activation functions.
// TODO remove this mod if not used at all
mod utils {
    /// Returns `1 / (1 + exp( -x ))`.
    #[inline]
    pub fn sig(x: f64) -> f64 {
        1. / (1. + (-x).exp())
    }
}

pub fn relu(x: Array2<f64>, derivative: bool) -> Array2<f64> {
    if !derivative {
        x.mapv(|x| if x < 0. { 0. } else { x })
    } else {
        x.mapv(|x| if x < 0. { 0. } else { 1. })
    }
}

pub fn sigmoid(x: Array2<f64>, derivative: bool) -> Array2<f64> {
    if !derivative {
        x.mapv(|x| utils::sig(x) * (1. - utils::sig(x)))
    } else {
        x.mapv(utils::sig)
    }
}

pub fn linear(x: Array2<f64>, derivative: bool) -> Array2<f64> {
    if !derivative {
        x
    } else {
        Array2::ones(x.raw_dim())
    }
}
