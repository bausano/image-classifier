use rand::prelude::*;
use super::layer::Layer;
use super::activation::Activation;

pub struct Network {

  // Collection of layers in the network. First layer is input layer, therefore
  // its weights do not matter. The last layer is an output layer. A classifier
  // should have as many output neurons in last layer as there are categories.
  // Minimal length of the vector is 2 (meaning there is no hidden layer).
  pub layers: Vec<Layer>,

  // Activation function and its derivative.
  pub activation: Activation,

  // Learning rate of the network.
  pub learning_rate: f64,

}

impl Network {

  // Builds new network instance from given activation function and raw layer
  // vectors in format Vector<(Layer): Vector<(Neuron) bias, weights>>
  //
  // @param activation Activation function and its derivative
  // @param layers Raw layers vector
  pub fn from (
    activation: Activation,
    layers: Vec<Vec<(f64, Vec<f64>)>>,
  ) -> Self {
    Network {
      learning_rate: 0.2_f64,
      activation,
      layers: layers.into_iter()
        .map(|neurons| Layer::from(neurons))
        .collect()
    }
  }

  // Generates new network from given schema. Schema is a vector of integers
  // where each integer represents one layer and the value of the integer number
  // of neurons there should be.
  //
  // @param activation Pointer to the activation function and its derivative
  // @param schema Vector representing layers and their neurons
  // @return New Network instance
  pub fn new (activation: Activation, schema: Vec<u8>) -> Self {
    // Random number generator.
    let mut rng = rand::thread_rng();

    // We prepare an empty layers shell.
    let mut layers: Vec<Vec<(f64, Vec<f64>)>> = Vec::new();

    // We don't have to construct the first layer, so we start our iterator
    // from the second layer.
    for i in 1..schema.len() {
      let mut layer: Vec<(f64, Vec<f64>)> = Vec::new();

      // Building each neuron for layer.
      for _ in 0..schema[i] {
        layer.push((
          // It does not matter what we initialize the bias to be.
          0_f64,
          // We have to create weights that will match number of inputs from
          // previous layer, which is the previous layer in the schema.
          (0..schema[i - 1])
            // We randomly assign each weight.
            .map(|_| rng.gen::<f64>() * 2_f64 - 1_f64)
            .collect()
        ));
      }

      layers.push(layer);
    }

    // Use the randomly generated skelet to build the network.
    Network::from(activation, layers)
  }

}
