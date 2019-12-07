extern crate ndarray;
extern crate spitz;

#[test]
fn init_network() {
    let mut arch = Vec::new();
    arch.push(spitz::Layer::new(4, 3, "relu"));
    arch.push(spitz::Layer::new(3, 5, "sigmoid"));
    let network = spitz::NNetwork::new(arch);
    let weights = network.weights;
    for element in weights {
        println!("{:8.4}", element);
    }
    //assert_eq!(1, 2);
}
