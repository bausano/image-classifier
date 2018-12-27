
pub struct Layer {
  pub neurons: Vec<(f64, Vec<f64>)>,
  pub outputs: Vec<f64>
}

impl Layer {
  pub fn from(neurons: Vec<(f64, Vec<f64>)>) -> Layer {
    Layer {
      neurons,
      outputs: Vec::new()
    }
  }

  pub fn process(&self, inputs: Vec<f64>) -> Vec<f64> {
    self.neurons.iter()
      .map(|neuron| {
        let (bias, weights) = neuron;

        product(&weights, &inputs) + bias
      })
      .map(|threshold| 0_f64.max(threshold))
      .collect()
  }
}

fn product(weights: &Vec<f64>, inputs: &Vec<f64>) -> f64 {
  let mut product = 0_f64;

  for i in 0..weights.len() {
    product += weights[i] * inputs[i];
  }

  product
}
