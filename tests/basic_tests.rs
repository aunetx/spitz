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
fn import_datas() {
    let x = &array![[0., 1., 2.], [1., 2., 3.], [2., 3., 4.], [3., 4., 5.]];
    let y = &array![[0.], [1.], [2.], [3.]];

    let mut network = NNetwork::new();
    network.import_datas(x, y, Some(0.25));

    assert!(network.get_is_test());
    assert_eq!(network.datas.test_x, array![[3., 4., 5.]]);
    assert_eq!(network.datas.test_y, array![[3.]]);
    assert_eq!(
        network.datas.train_x,
        array![[0., 1., 2.], [1., 2., 3.], [2., 3., 4.]]
    );
    assert_eq!(network.datas.train_y, array![[0.], [1.], [2.]]);
}

#[test]
fn import_datas_no_test() {
    let x = &array![[0., 1., 2.], [1., 2., 3.], [2., 3., 4.], [3., 4., 5.]];
    let y = &array![[0.], [1.], [2.], [3.]];

    let mut network = NNetwork::new();
    network.import_datas(x, y, None);

    assert!(!network.get_is_test());
    assert_eq!(network.datas.test_x, array![[]]);
    assert_eq!(network.datas.test_y, array![[]]);
    assert_eq!(&network.datas.train_x, x);
    assert_eq!(&network.datas.train_y, y);
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

    network.set_epochs(15);
    assert_eq!(network.epochs, 15);
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

#[test]
fn train_xor() {
    simple_logger::init_with_level(log::Level::Info).unwrap();
    log::info!("Begun");
    let x = &array![[0., 0.], [0., 1.], [1., 0.], [1., 1.],];
    let y = &array![[0.], [1.], [1.], [0.]];

    let mut network = NNetwork::new();
    let _pred = network
        .import_datas(x, y, None)
        .input_layer(2)
        .add_layer(4, Activation::Relu)
        .add_layer(5, Activation::Sigmoid)
        .add_layer(1, Activation::Sigmoid)
        .set_learning_rate(0.3)
        .set_epochs(1500)
        .init()
        .fit()
        .feed_forward(&array![[0., 1.]])
        .last()
        .unwrap()
        .clone();

    log::info!("Done")
    //assert_eq!(array![[1.]], pred);
}
