extern crate mnist_extractor;
extern crate simple_logger;
use ndarray::prelude::*;
use spitz::*;

// Init logger : prevents logger from being initialized twice.
use std::sync::Once;
static INIT: Once = Once::new();
/// Setup function that is only run once, even if called multiple times.
fn setup() {
    INIT.call_once(|| {
        simple_logger::init_with_level(log::Level::Debug).unwrap();
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
        .set_batches(1)
        .init()
        .fit()
        .feed_forward(x)
        .last()
        .unwrap()
        .clone();

    log::info!("Done");
}

pub fn show_image(imgs: &Array2<f64>, img_to_show: usize) {
    for (id, &el) in imgs.row(img_to_show).iter().enumerate() {
        if id % 28 == 0 {
            println!();
        } else if el == 0. {
            print!("   ");
        } else if el < 0.25 {
            print!(" . ");
        } else if el < 0.5 {
            print!(" * ");
        } else if el < 0.75 {
            print!("***");
        } else {
            print!("&&&")
        }
    }
    println!();
}

#[test]
fn train_mnist() {
    setup();
    log::warn!("Init");

    let (test_labels, test_images, train_labels, train_images) = &mnist_extractor::get_all();

    //show_image(test_images, 1);

    log::warn!("Begun");

    let mut network = NNetwork::new();
    let pred = network
        .import_train_datas(train_images, train_labels)
        .import_test_datas(test_images, test_labels)
        .input_layer(784)
        .add_layer(10, Activation::Sigmoid)
        .set_learning_rate(0.03)
        .set_epochs(5)
        .set_batches(10)
        .init()
        .fit()
        .feed_forward(test_images)
        .last()
        .unwrap()
        .clone();

    log::debug!("PRED = {:8.4}", pred);

    log::warn!("Done");
}
