use std::ops::Deref;
use super::network::Network;

impl Network {

  // Trains the network with back prop algorithm.
  //
  // @param network Network instance we want to train
  // @param training_data Training data
  pub fn train (&mut self, training_data: Vec<(u8, Vec<f64>)>) {

    for digit in training_data {
      let (target, inputs) = digit;

      // Gets the activations for each layer.
      let activations = self.calculate_activations(inputs);

      // Calculates the errors of each output neuron.
      let cost = self.calculate_cost(target, activations.last().unwrap());
    }

  }

  // Computes the activation of the network over given inputs and stores them
  // along the way in a vector. In contrast with network_classify compute fn,
  // we have activations from all layers, not only the last one.
  //
  // @param inputs Vector of same length as input layer
  // @return Activation intensity of each neuron in each layer
  fn calculate_activations (&self, inputs: Vec<f64>) -> Vec<Vec<f64>> {
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

  // Calculates the cost (error) of the classification.
  //
  // @param target The expected outcome
  // @param outputs Outputs from the network
  // @return Vector of errors for each output neuron
  fn calculate_cost (&self, target: u8, outputs: &Vec<f64>) -> Vec<f64> {
    // Converts target into usize so that it can be compared with enumerate.
    let target = usize::from(target);

    outputs.iter()
      .enumerate()
      .map(|(neuron, output)| {
        // If this was the expected answer, compute error to 1, otherwise to 0.
        if target == neuron {
          1_f64 - output
        } else {
          0_f64 - output
        }
      })
      // Make the error more forgiving. This is better so that the network
      // doesn't make too big steps.
      .map(|activation| activation.powi(2))
      .collect()
  }

}
