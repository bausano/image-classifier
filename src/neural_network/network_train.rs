use std::ops::Deref;
use super::layer::Layer;
use super::network::Network;

impl Network {

  // Trains the network with back prop algorithm.
  //
  // @param network Network instance we want to train
  // @param training_data Training data
  pub fn train (&mut self, training_data: Vec<(u8, Vec<f64>)>) {

    for digit in training_data {
      let (target, inputs) = digit;

      // Converts target into usize so that it can be compared with enumerate.
      let target = usize::from(target);

      // Gets the activations for each layer.
      let activations: Vec<Vec<f64>> = self.calculate_activations(inputs);

      let layers_count: usize = self.layers.len() - 1;

      for layer in layers_count..1 {
        if (layer == layers_count) {
          self.output_layer_weights(
            layer,
            target,
            activations,
          );
        }
      }
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

  // Calculates output layer weights. This process is slightly different for
  // this layer, therefore it has its own logic separated from the hidden layers.
  //
  // @param layer_index Index of the output layer is layers.length - 1
  // @param target The expected result for given inputs
  // @param activations Activation values of each layer
  // @return Vector of changes to each neurons bias and weights
  fn output_layer_weights (
    &self,
    layer_index: usize,
    target: usize,
    activations: &Vec<Vec<f64>>,
  ) -> Vec<(f64, Vec<f64>)> {
    // Partial weight change without the
    let partial_deltas: Vec<f64> = self.calculate_deltas(
      target,
      &activations[layer_index]
    );

    // Finishes deltas for each neuron in the output layer.
    partial_deltas.iter().enumerate()
      .map(|(neuron, partial_delta)| {
        // Get the neuron this delta is computed for and extract the bias.
        let (bias, _) = self.layers[layer_index].neurons[neuron];

        // The partial delta onto the activations from the previous layer to
        // find the final delta to the weight.
        let weight_deltas = activations[layer_index - 1].iter()
          .map(|activation| activation * partial_delta)
          .collect();

        // Export changes to the neuron in the same format as each neuron is
        // defined: (bias, weights).
        (bias * partial_delta, weight_deltas)
      })
      .collect()
  }

  // Calculates the the partial weight change for each output neuron. This
  // result is to be mapped over the outputs in previous hidden layer.
  //
  // @param target The expected outcome
  // @param outputs Outputs from the network
  // @return Vector of partial delta for each output neuron
  fn calculate_deltas (&self, target: usize, outputs: &Vec<f64>) -> Vec<f64> {
    let derivative = self.activation.derivative.deref();

    outputs.iter()
      .enumerate()
      .map(|(neuron, output)| {
        // If this was the expected answer, compute error to 1, otherwise to 0.
        let total_to_output = if target == neuron {
          1_f64 - output
        } else {
          0_f64 - output
        };

        self.learning_rate * derivative(output.clone()) * total_to_output
      })
      .collect()
  }
}
