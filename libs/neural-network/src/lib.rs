/*
A simple feedforward implementation of a neural network.
Each neuron uses the ReLU activation function (can be changed later).
Given some Layer-Topology, the network is constructed with random weights and biases and fully connected.
*/


use rand::Rng;




#[derive(Debug)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

#[derive(Debug)]
struct Layer {
    neurons: Vec<Neuron>,
}

#[derive(Debug)]
struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

#[derive(Debug)]
pub struct LayerTopology {
    pub neurons: usize,
}


impl Layer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut neurons = Vec::with_capacity(output_size);
        for _ in 0..output_size {
            neurons.push(Neuron::new(input_size));
        }
        Self { neurons }
    }

    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        let mut outputs = Vec::new();
        for neuron in &self.neurons {
            let output = neuron.propagate(&inputs);
            outputs.push(output);
        }
        outputs
    }

    fn add_neuron(&mut self, neuron: Neuron) {
        self.neurons.push(neuron);
    }
}


impl Neuron {
    fn new(input_size: usize) -> Self {
        let mut rng = rand::rng();
        let bias = rng.random_range(-1.0..=1.0);
        let weights = (0..input_size).map(|_| rng.random_range(-1.0..=1.0)).collect();
        Self { bias, weights }
    }
    
    fn propagate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());

        let mut output = 0.0;

        for i in 0..inputs.len() {
            output += inputs[i] * self.weights[i];
        }

        output += self.bias;

        if output > 0.0 {
            output} else {
                0.0
            }
        }
}



impl NeuralNetwork {
    pub fn new(layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1, "Network must have at least input and output layers");
        let mut network_layers = Vec::new();

        for i in 0..layers.len() - 1 {
            let input_size = layers[i].neurons;
            let output_size = layers[i + 1].neurons;

            network_layers.push(Layer::new(input_size, output_size));
            
        }

        Self { layers: network_layers }
    }


    pub fn propagate(&self, mut inputs: Vec<f32>) -> Vec<f32> {
        for layer in &self.layers {
            inputs = layer.propagate(inputs);
        }
        inputs
    }

    pub fn mutate(&mut self, mutation_rate: f32, mutation_amount: f32) {
        let mut rng = rand::rng();

        for layer in &mut self.layers {
            for neuron in &mut layer.neurons {
                if rng.random::<f32>() < mutation_rate {
                    neuron.bias += rng.random_range(-mutation_amount..=mutation_amount);
                }
                for weight in &mut neuron.weights {
                    if rng.random::<f32>() < mutation_rate {
                        *weight += rng.random_range(-mutation_amount..=mutation_amount);
                    }
                }
            }
        }
    }

    pub fn print_structure(&self) {
        for (i, layer) in self.layers.iter().enumerate() {
            println!("Layer {}: {} neurons", i, layer.neurons.len());
            for (j, neuron) in layer.neurons.iter().enumerate() {
                println!("  Neuron {}: bias = {}, weights = {:?}", j, neuron.bias, neuron.weights);
            }
        }
    }

    pub fn add_layer(&mut self, position: usize, topology: &LayerTopology) {
        assert!(position > 0 && position < self.layers.len(), "Position out of bounds");

        let input_size = self.layers[position - 1].neurons.len();
        let output_size = topology.neurons;

        let new_layer = Layer::new(input_size, output_size);

        self.layers.insert(position, new_layer);

        let next_layer_input_size = output_size;
        let next_layer_output_size = self.layers[position + 1].neurons.len();

        let mut adjusted_next_layer = Layer::new(next_layer_input_size, next_layer_output_size);

        for _neuron in &self.layers[position + 1].neurons {
            adjusted_next_layer.add_neuron(Neuron::new(next_layer_input_size));
        }

        self.layers[position + 1] = adjusted_next_layer;
    }

}

impl Clone for NeuralNetwork {
    fn clone(&self) -> Self {
        let mut cloned_layers = Vec::new();
        for layer in &self.layers {
            let mut cloned_neurons = Vec::new();
            for neuron in &layer.neurons {
                cloned_neurons.push(Neuron {
                    bias: neuron.bias,
                    weights: neuron.weights.clone(),
                });
            }
            cloned_layers.push(Layer {
                neurons: cloned_neurons,
            });
        }
        Self {
            layers: cloned_layers,
        }
    }
}
