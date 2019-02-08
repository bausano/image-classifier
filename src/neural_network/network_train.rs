use std::ops::Deref;
use super::network::Network;

impl Network {

  // Trains the network with back prop algorithm.
  //
  // @param network Network instance we want to train
  // @param training_data Training data
  pub fn train(&mut self, training_data: Vec<(u8, Vec<f64>)>) {

    for digit in training_data {
      let (target, inputs) = digit;

      let activations = self.calculate_activations(inputs);
    }

  }

  // Computes the activation of the network over given inputs and stores them
  // along the way in a vector. In contrast with network_classify compute fn,
  // we have activations from all layers, not only the last one.
  //
  // @param inputs Vector of same length as input layer
  // @return Activation intensity of each neuron in each layer
  fn calculate_activations(&self, inputs: Vec<f64>) -> Vec<Vec<f64>> {
    // We deference the pointer to the activation function.
    let activation_fn = self.activation.function.deref();

    // We return vector of vectors holding the activation values.
    self.layers.iter().fold(vec!(inputs), |mut activations, layer| {
      // Computing an activation vector of a layer.
      let output = layer.activations(
        &activations.last().unwrap(),
        activation_fn,
      );

      // Propagating it forward.
      activations.push(output);

      activations
    })
  }

}
