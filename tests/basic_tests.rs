extern crate ndarray;
extern crate spitz;
use ndarray::prelude::*;
use spitz::*;

#[test]
fn init_architecture() {
    let mut network = NNetwork::new();
    let arch_list = [10, 40, 5].to_vec();

    network.init_architecture(arch_list);

    for layer in &network.architecture {
        println!("{:?}", layer);
    }
    //assert_eq!(1, 2);
}

#[test]
fn init_weights() {
    let mut network = NNetwork::new();
    let arch_list = [10, 40, 5].to_vec();

    network.init_architecture(arch_list);
    network.init_weights();

    assert_eq!(network.weights[0].shape(), &[10, 40]);
    assert_eq!(network.weights[1].shape(), &[40, 5]);
}

#[test]
fn import_datas() {
    let x = array![[0., 1., 2.], [1., 2., 3.], [2., 3., 4.], [3., 4., 5.]];
    let y = array![[0.], [1.], [2.], [3.]];

    let mut network = NNetwork::new();
    network.import_datas(x, y, Some(0.25));

    assert!(network.is_test);
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
    let x = array![[0., 1., 2.], [1., 2., 3.], [2., 3., 4.], [3., 4., 5.]];
    let y = array![[0.], [1.], [2.], [3.]];

    let mut network = NNetwork::new();
    network.import_datas(x.clone(), y.clone(), None);

    assert!(!network.is_test);
    assert_eq!(network.datas.test_x, array![[]]);
    assert_eq!(network.datas.test_y, array![[]]);
    assert_eq!(network.datas.train_x, x);
    assert_eq!(network.datas.train_y, y);
}

#[test]
#[allow(clippy::float_cmp)]
fn set_learning_rate() {
    let mut network = NNetwork::new();

    network.learning_rate(0.1);
    assert_eq!(network.learning_rate, 0.1);
}

#[test]
fn test_relu() {
    let x: Array2<f64> = array![[-5., 8., -6., 0.], [2., 0., -1., 105.]];
    assert_eq!(
        array![[0., 8., 0., 0.], [2., 0., 0., 105.]],
        activations::relu(x.clone(), false)
    );
    assert_eq!(
        array![[0., 1., 0., 1.], [1., 1., 0., 1.]],
        activations::relu(x, true)
    );
}

#[test]
fn train_xor() {
    let x = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.],];
    let y = array![[0.], [1.], [1.], [0.]];

    let mut network = NNetwork::new();
    network.import_datas(x, y, None);

    assert_eq!(1, 1);
}
