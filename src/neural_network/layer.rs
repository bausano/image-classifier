
/// Network layer carries vector of neuron and performs computations on them.
pub struct Layer {

  /// Vector of neurons with associated weights and bias weight. Each weight
  /// is semantically connected to one neuron in previous layer. The bias weight
  /// is always on maximum input, 1, therefore we don't have to keep a connection
  /// to a bias neuron from previous layer.
  ///
  /// Following vector is therefore in following format:
  /// neurons: Vector<(Neuron) bias, weights>
  ///
  /// At the end of the training process, we change the floats.
  pub neurons: Vec<(f64, Vec<f64>)>,

  /// Shadow vector we can store weights updates to. The first number states how
  /// many updates there has been since last commit. This is used to divide each
  /// of the weight update before addding it to the neurons weights.
  updates: (f64, Vec<(f64, Vec<f64>)>),

}

impl Layer {

  /// Layer builder that is used to bootstrap the layer.
  ///
  /// @param neurons Vector of neurons in the layer
  /// @return New layer instance
  pub fn from (neurons: Vec<(f64, Vec<f64>)>) -> Layer {
    Layer {
      updates: Layer::new_updates(&neurons),
      neurons,
    }
  }

  /// Calculates the activations for each neuron against inputs. The network
  /// builder ensures that there is going to be same number of weights as inputs.
  ///
  /// @param inputs Activations from previous layer
  /// @param activation_function Function of x which puts the output in range
  /// @return Activations for this layer
  pub fn activations (
    &self,
    inputs: &Vec<f64>,
    activation_function: &Fn(f64) -> f64,
  ) -> Vec<f64> {
    self.neurons.iter()
      .map(|neuron| {
        // Destruct each neuron into its weights and bias.
        let (bias, weights) = neuron;

        let mut product = 0_f64;
        // Matrix multiplication of weights and inputs.
        for i in 0..weights.len() {
          product += weights[i] * inputs[i];
        }

        // Bias has always input 1, therefore we can just add the bias weight.
        product + bias
      })
      // We map every neurons activation with given activation function.
      .map(activation_function)
      .collect()
  }

  /// Adds new weights to the cached update vector of neurons.
  ///
  /// @param update Vector that mimics neurons with nudges to the weights
  pub fn add_update (&mut self, update: Vec<(f64, Vec<f64>)>) {
    // Add weights to the update vector.
    Layer::add_weights(&mut self.updates.1, &update, 1_f64);

    // Updates the number of commits.
    self.updates.0 = self.updates.0 + 1_f64;
  }

  /// Commits the cached update vector into the main neurons vector.
  pub fn commit_updates (&mut self) {
    {
      let (divider, ref mut neurons) = self.updates;

      // Adds weights to the neurons.
      Layer::add_weights(&mut self.neurons, neurons, -divider);
    }

    self.updates = Layer::new_updates(&self.neurons);
  }

  /// Adds weights to a vector. This is used by commit and add methods.
  ///
  /// @param targets Vector of target neurons
  /// @param source Vector of nudges to weights and biases
  /// @param divider Number to divide each source weight and bias by
  fn add_weights (
    targets: &mut Vec<(f64, Vec<f64>)>,
    source: &Vec<(f64, Vec<f64>)>,
    divider: f64,
  ) {
    if divider == 0_f64 {
      return
    }

    // For each element in the target vector.
    for (index, target) in targets.iter_mut().enumerate() {
      let (bias, ref weights) = source[index];

      // Add bias from source to the target.
      target.0 = target.0 + (bias / divider);

      for (weight_index, weight) in weights.iter().enumerate() {
        // Add weight from source to the target.
        let new_weight = target.1[weight_index] + (weight / divider);

        // Change the weight to the summed one.
        target.1[weight_index] = new_weight;
      }
    }
  }

  /// Generates new vector of neurons with weights and biases 0 and a counter
  /// starting at 0 for the update vector.
  ///
  /// @param neurons Scheme of neurons
  /// @return Counter and copy of neurons
  fn new_updates (neurons: &Vec<(f64, Vec<f64>)>) -> (f64, Vec<(f64, Vec<f64>)>) {
    (
      0_f64,
      // Sets bias and all weights to 0 for each neuron.
      (0..neurons.len())
        .map(|neuron_index| {
          let (_, ref weights) = neurons[neuron_index];

          (0_f64, (0..weights.len()).map(|_| 0_f64).collect())
        })
        .collect()
    )
  }
}
