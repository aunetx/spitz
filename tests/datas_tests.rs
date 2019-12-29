extern crate simple_logger;
use ndarray::prelude::*;
use spitz::*;

#[test]
fn import_datas() {
    let x = &array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.], [9., 10., 11.]];
    let y = &array![[0.], [1.], [2.], [3.]];

    let mut network = NNetwork::new();
    network.import_datas(x, y, 0.25).init();

    let mut wanted_datas_train: Vec<DatasTrain> = Vec::new();
    wanted_datas_train.push(DatasTrain {
        x: array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        y: array![[0.], [1.], [2.]],
    });

    assert_eq!(network.datas.test.x, array![[9., 10., 11.]]);
    assert_eq!(network.datas.test.y, array![[3.]]);
    assert_eq!(network.datas.train, wanted_datas_train);
}

#[test]
fn import_datas_no_test() {
    let x = &array![[0., 1., 2.], [1., 2., 3.], [2., 3., 4.], [3., 4., 5.]];
    let y = &array![[0.], [1.], [2.], [3.]];

    let mut network = NNetwork::new();
    network.import_train_datas(x, y).init();

    let mut wanted_datas_train: Vec<DatasTrain> = Vec::new();
    wanted_datas_train.push(DatasTrain {
        x: array![[0., 1., 2.], [1., 2., 3.], [2., 3., 4.], [3., 4., 5.]],
        y: array![[0.], [1.], [2.], [3.]],
    });

    // TODO verify which shape should be used if there is no test data for `Datas.test.{x, y}`
    //assert_eq!(network.datas.test.x, array![[]]);
    //assert_eq!(network.datas.test.y, array![[]]);
    assert_eq!(network.datas.train, wanted_datas_train);
}

#[test]
fn import_datas_batches() {
    let x = &array![[0., 1., 2.], [1., 2., 3.], [2., 3., 4.], [3., 4., 5.]];
    let y = &array![[0.], [1.], [2.], [3.]];

    let mut network = NNetwork::new();
    network.import_train_datas(x, y).set_batches(1).init();

    let mut wanted_datas_train: Vec<DatasTrain> = Vec::new();
    wanted_datas_train.push(DatasTrain {
        x: array![[0., 1., 2.]],
        y: array![[0.]],
    });
    wanted_datas_train.push(DatasTrain {
        x: array![[1., 2., 3.]],
        y: array![[1.]],
    });
    wanted_datas_train.push(DatasTrain {
        x: array![[2., 3., 4.]],
        y: array![[2.]],
    });
    wanted_datas_train.push(DatasTrain {
        x: array![[3., 4., 5.]],
        y: array![[3.]],
    });

    assert_eq!(network.datas.train, wanted_datas_train);
}
