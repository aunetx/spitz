use ndarray::prelude::Array2;

/// Utilitaries for activation functions.
mod utils {
    /// Returns `x` if `x > 0`, else `0`.
    #[inline]
    pub fn max(x: f64) -> f64 {
        if x < 0. {
            0.
        } else {
            x
        }
    }

    /// Returns `1` if `x > 0`, else `0`.
    #[inline]
    pub fn sup(x: f64) -> f64 {
        if x < 0. {
            0.
        } else {
            1.
        }
    }

    /// Returns `1 / (1 + exp( -x ))`.
    #[inline]
    pub fn sig(a: f64) -> f64 {
        1. / (1. + (-a).exp())
    }
}

pub fn relu(x: Array2<f64>, derivative: bool) -> Array2<f64> {
    if !derivative {
        x.mapv(utils::max)
    } else {
        x.mapv(utils::sup)
    }
}

pub fn sigmoid(x: Array2<f64>, derivative: bool) -> Array2<f64> {
    if !derivative {
        x.mapv(|a| utils::sig(a) * (1. - utils::sig(a)))
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
