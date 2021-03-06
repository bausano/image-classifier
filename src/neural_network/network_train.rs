use std::ops::Deref;
use super::layer::Layer;
use super::network::Network;

impl Network {

  /// Trains the network with back prop algorithm.
  ///
  /// @param network Network instance we want to train
  /// @param training_data Training data
  pub fn train (&mut self, training_data: &Vec<(u8, Vec<f64>)>, epoch: usize) {
    self.learning_rate = self.calculate_learning_rate(epoch as f64);

    for (i, digit) in training_data.iter().enumerate() {
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

      let batch = i % self.batch_size;

      if batch % self.batch_size == 0 {
        for layer in self.layers.iter_mut() {
          layer.commit_updates();
        }
      }
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

  /// Calculates the nudges to bias and neurons. This is the core of the whole
  /// algorithm. It also pushes the calculated new partial deltas to a
  /// collector that sends it to the next layer.
  ///
  /// @param layer_index Current layer
  /// @param partial_deltas Errors from the previous layer
  /// @param activations All network activations from the feed forward process
  /// @param new_partial_deltas Delta error collector
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
        let new_partial_delta = if layer_index == self.layers.len() - 1 {
          // We already have calculated deltas for the output layer, so we skip
          // the process of calculating them.
          partial_deltas[neuron_index]
        } else {
          // We use neurons from the previous layer because that's where the
          // weights that participate to the total error are.
          let neurons = &self.layers[layer_index + 1].neurons;
          // Derivative of activation of current neuron.
          let activation_derivative = derivative(activations[layer_index + 1][neuron_index]);

          // Multiply the derivative and sum of errors multiplied by relevant
          // weight.
          activation_derivative * partial_deltas.iter().enumerate()
            .fold(0_f64, |sum, (delta_index, delta)| {
              let (_, ref weights) = neurons[delta_index];

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

  pub fn calculate_learning_rate (&mut self, epoch: f64) -> f64 {
    let step = 1_f64 + epoch / (2_f64 * self.step_size as f64);
    let cycle = step.floor();
    let progress = (0.5_f64 - (step - cycle)).abs();

    self.max_lr - (
      2_f64 * (self.max_lr - self.min_lr) * (0.5_f64 - progress).abs()
    )
  }

}
