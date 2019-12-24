extern crate mnist_extractor;
extern crate ndarray;
extern crate simple_logger;
extern crate spitz;
use ndarray::prelude::*;
use spitz::*;

// Init logger : prevents logger from being initialized twice.
use std::sync::Once;
static INIT: Once = Once::new();
/// Setup function that is only run once, even if called multiple times.
fn setup() {
    INIT.call_once(|| {
        simple_logger::init_with_level(log::Level::Warn).unwrap();
    });
}

// ! XOR LEARNING TEST ---------------
#[test]
fn train_xor() {
    setup();
    log::info!("Begun");
    let x = &array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let y = &array![[0.], [1.], [1.], [0.]];

    let mut network = NNetwork::new();
    let _pred = network
        .import_train_datas(x, y)
        .input_layer(2)
        .add_layer(4, Activation::Relu)
        .add_layer(5, Activation::Sigmoid)
        .add_layer(1, Activation::Sigmoid)
        .set_learning_rate(0.3)
        .set_epochs(10)
        .init()
        .print_weights()
        .fit()
        .print_weights()
        .feed_forward(x)
        .last()
        .unwrap()
        .clone();

    log::info!("Done");
    //assert_eq!(*y, _pred);
}

#[test]
fn train_mnist() {
    setup();
    log::warn!("Init");
    let (test_labels, test_images, train_labels, train_images) = mnist_extractor::get_all();

    log::warn!("Begun");

    let mut network = NNetwork::new();
    let pred = network
        .import_train_datas(&train_images, &train_labels)
        .import_test_datas(&test_images, &test_labels)
        .input_layer(784)
        .add_layer(10, Activation::Linear)
        .set_learning_rate(0.03)
        .set_epochs(5)
        .init()
        .fit()
        .feed_forward(&test_images)
        .last()
        .unwrap()
        .clone();

    log::debug!("PRED = {:8.4}", pred);

    log::warn!("Done");
}