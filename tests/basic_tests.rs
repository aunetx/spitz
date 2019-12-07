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
