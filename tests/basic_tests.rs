extern crate ndarray;
extern crate simple_logger;
extern crate spitz;
use ndarray::prelude::*;
use spitz::*;

#[test]
fn add_layer() {
    let mut network = NNetwork::new();

    network
        .input_layer(10)
        .add_layer(50, Activation::Relu)
        .add_layer(40, Activation::Sigmoid)
        .init();

    for layer in network.get_architecture().layers {
        println!("{:?}", layer);
    }
}

#[test]
fn init_weights() {
    let mut network = NNetwork::new();

    network
        .input_layer(10)
        .add_layer(40, Activation::Relu)
        .add_layer(5, Activation::Sigmoid)
        .init();
    let weights = network.get_weights();

    assert_eq!(weights[0].shape(), &[10, 40]);
    assert_eq!(weights[1].shape(), &[40, 5]);
}

#[test]
#[allow(clippy::float_cmp)]
fn set_learning_rate() {
    let mut network = NNetwork::new();

    network.set_learning_rate(0.1);
    assert_eq!(network.learning_rate, 0.1);
}

#[test]
fn set_epochs() {
    let mut network = NNetwork::new();

    network.set_epochs(25);
    assert_eq!(network.epochs, 25);
}

#[test]
fn set_batches() {
    let mut network = NNetwork::new();

    network.set_batches(25);
    assert_eq!(network.batches, 25);
}

#[test]
fn test_relu() {
    let x: Array2<f64> = array![[-5., 8., -6., 0.], [2., 0., -1., 105.]];
    assert_eq!(
        array![[0., 8., 0., 0.], [2., 0., 0., 105.]],
        maths::activations::relu(x.clone(), false)
    );
    assert_eq!(
        array![[0., 1., 0., 1.], [1., 1., 0., 1.]],
        maths::activations::relu(x, true)
    );
}
