# spitz

## Getting started

### Description

**IMPORTANT : this project should not be used in production.
It does not even train well yet.**

This project is a library that provides a pure rust deep learning implementation.\
Its objectives are :

- ease of use :
*initialize and fit a network painless*

- performances :
*provide nearly-instantaneous predictions, and a very quick learning process*

- portability :
*use as few external libraries as possible*

- adaptability :
*use the same learning process for fairly differents objectives*

### Prerequesites

Just a recent rust compilater. Its dependencies [`ndarray`](https://docs.rs/ndarray/0.12.1/ndarray/), [`ndarray-rand`](https://github.com/rust-ndarray/ndarray/tree/master/ndarray-rand) and [`log`](https://docs.rs/log/0.4.6/log/) will be automatically imported during compilation.

Also, for the moment you need to have the package [`libopenblas-dev`](https://www.openblas.net/) and its dependencies installed on your computer : it provides really fast matrix operations. It will later be optionnal, although recommended.
Under `ubuntu` and its derivatives : `apt install libopenblas-base libopenblas-dev gfortran`.

### Importation

To import `spitz` in your project, add in your `Cargo.toml` file :

```toml
[dependencies]
spitz = { git = "https://github.com/aunetx/spitz" }
```

And in your code :

```rust
extern crate spitz;
use spitz::*;
```

## Utilisation

### Train the network

Once `spitz` is imported in your project, you can first create a new `NNetwork` variable.

```rust
let mut network = NNetwork::new();
```

Then use the different methods provided to use it as you want to :

```rust
// some datas used for training, here : XOR
let x = &array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
let y = &array![[0.], [1.], [1.], [0.]];

// import datas : two ndarray's Array2,
// one represents the inputs : `x`,
// the other one the outputs : `y`.
// the third param is an option that contains the ratio of test datas to pick.
network.import_datas(x, y, None);

// set the size of input layer, then add hidden and output layers.
// also, select the activation to use (from Activation enum).
network.input_layer(2)
    .add_layer(4, Activation::Relu) // you can chain every calls !
    .add_layer(5, Activation::Sigmoid)
    .add_layer(1, Activation::Sigmoid)

    // set hyperparameters (if not, defaults are used)
    .set_learning_rate(0.3) // default = 0.03
    .set_epochs(1500) // default = 15

// call init before fitting the network
    .init()
    .fit();
```

#### Logging

If you want to have outputs during the training process, you should use a logger such as [`simple_logger`](https://github.com/borntyping/rust-simple_logger) :

```rust
simple_logger::init().unwrap();
// your code here...
```

## Contributing

Please feel *free* to contribute to that project, fork it, clone it, make it suffer, do whatever you want actually.
And if you find bugs, please open an issue and make pull requests if you want to.

Sincerely yours,
-- aunetx --
