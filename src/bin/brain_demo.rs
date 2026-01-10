use predatorVsPrey::brain_neural_network::NeuralNetwork;
use predatorVsPrey::settings;

use rand::thread_rng;
use std::fs;

fn main() {
    let mut rng = thread_rng();

    let num_inputs = 6;
    let nn = NeuralNetwork::new(num_inputs, 2, 10, settings::bias(), &mut rng);

    let dot = nn.to_dot();
    fs::write("neural_network.dot", dot).expect("failed to write neural_network.dot");

    println!("Wrote neural_network.dot");
    println!("If you have GraphViz installed, run:");
    println!("  dot -Tpng neural_network.dot -o neural_network.png");
}
