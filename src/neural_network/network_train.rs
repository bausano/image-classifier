use std::ops::Deref;
use super::layer::Layer;
use super::network::Network;

impl Network {

  /// Trains the network with back prop algorithm.
  ///
  /// @param network Network instance we want to train
  /// @param training_data Training data
  pub fn train (&mut self, training_data: &Vec<(u8, Vec<f64>)>) {
    for digit in training_data {
      let (target, inputs) = digit;

      // Converts target into usize so that it can be compared with enumerate.
      let target = usize::from(*target);

      // Gets the activations for each layer.
      let activations: Vec<Vec<f64>> = self.calculate_activations(inputs);

      let layers_count: usize = self.layers.len();

      // Partial weight change without the learning rate and previous
      // activations.
      let output_partial_deltas: Vec<f64> = self.calculate_deltas(
        target,
        &activations[layers_count],
      );

      let mut layers_iterator: Vec<usize> = (0..layers_count).collect();
      layers_iterator.reverse();

      // Propagates the error deltas from one layer to another.
      layers_iterator.iter().fold(
        output_partial_deltas,
        |deltas, layer| self.update_layer_weights(*layer, deltas, &activations),
      );
    }

    // Commits all updates into each layer.
    for layer in self.layers.iter_mut() {
      layer.commit_updates();
    }
  }

  /// Computes the activation of the network over given inputs and stores them
  /// along the way in a vector. In contrast with network_classify compute fn,
  /// we have activations from all layers, not only the last one.
  ///
  /// @param inputs Vector of same length as input layer
  /// @return Activation intensity of each neuron in each layer
  fn calculate_activations (&self, inputs: &Vec<f64>) -> Vec<Vec<f64>> {
    // We deference the pointer to the activation function.
    let activation_fn = self.activation.function.deref();

    // We return vector of vectors holding the activation values.
    self.layers.iter().fold(vec!(inputs.clone()), |mut activations, layer| {
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

  /// Calculates output layer weights. This process is slightly different for
  /// this layer, therefore it has its own logic separated from the hidden layers.
  ///
  /// @param layer_index Index of the output layer is layers.length - 1
  /// @param target The expected result for given inputs
  /// @param activations Activation values of each layer
  /// @return Vector of changes to each neurons bias and weights
  fn update_layer_weights (
    &mut self,
    layer_index: usize,
    partial_deltas: Vec<f64>,
    activations: &Vec<Vec<f64>>,
  ) -> Vec<f64> {
    // Stores delta errors for neurons in this layer.
    let mut new_partial_deltas: Vec<f64> = Vec::new();

    // Calculates nudges for each weight and bias in this layer.
    let nudges = self.calculate_nudges(
      layer_index,
      partial_deltas,
      activations,
      &mut new_partial_deltas,
    );

    // Caches the nudges.
    self.layers[layer_index].add_update(nudges);

    new_partial_deltas
  }

  fn calculate_nudges (
    &mut self,
    layer_index: usize,
    partial_deltas: Vec<f64>,
    activations: &Vec<Vec<f64>>,
    new_partial_deltas: &mut Vec<f64>,
  ) -> Vec<(f64, Vec<f64>)> {
    let layer: &Layer = &self.layers[layer_index];
    let derivative = self.activation.derivative.deref();

    layer.neurons.iter().enumerate()
      .map(|(neuron_index, _)| {
        // Calculates the error for current neuron as a product of its weights
        // and the error of neuron in the next layer that is that weight
        // connected to.
        let new_partial_delta = if layer_index == self.layers.len() - 1 {
          partial_deltas[neuron_index]
        } else {
          let neurons = &self.layers[layer_index + 1].neurons;
          let activation_derivative = derivative(activations[layer_index + 1][neuron_index]);

          activation_derivative * partial_deltas.iter().enumerate()
            .fold(0_f64, |sum, (delta_index, delta)| {
              // println!("{} neurons, {} delta, {} delta_index, {} neuron_undex", neurons.len(), delta, delta_index, neuron_index);
              let (_, ref weights) = neurons[delta_index];
              // println!("{} weights", weights.len());

              sum + (delta * weights[neuron_index])
            })
        };

        // Pushes this partial delta to output vector that is going to be passed
        // to next layer.
        new_partial_deltas.push(new_partial_delta);

        // The partial delta onto the activations from the previous layer to
        // find the final delta to the weight.
        // In the activations vector, activations indecies are equal to
        // layer_index - 1, because the layers vector does not include first
        // input layer. Therefore activations[layer_index] gives us activations
        // from the next layer (in direction to the output).
        let weight_deltas = activations[layer_index].iter()
          .map(|activation| activation * new_partial_delta * self.learning_rate)
          .collect();

        // Export changes to the neuron in the same format as each neuron is
        // defined: (bias, weights).
        (new_partial_delta * self.learning_rate, weight_deltas)
      })
      .collect()
  }

  /// Calculates the the partial weight change for each output neuron. This
  /// result is to be mapped over the outputs in previous hidden layer.
  ///
  /// @param target The expected outcome
  /// @param outputs Outputs from the network
  /// @return Vector of partial delta for each output neuron
  fn calculate_deltas (&self, target: usize, outputs: &Vec<f64>) -> Vec<f64> {
    let derivative = self.activation.derivative.deref();

    outputs.iter()
      .enumerate()
      .map(|(neuron, output)| {
        // If this was the expected answer, compute error to 1, otherwise to 0.
        let total_to_output = if target == neuron {
          -(1_f64 - output)
        } else {
          -(0_f64 - output)
        };

        derivative(*output) * total_to_output
      })
      .collect()
  }
}
