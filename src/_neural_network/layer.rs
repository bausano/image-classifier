
pub struct Layer {
  pub neurons: Vec<(f64, Vec<f64>)>
}

impl Layer {
  pub fn from(neurons: Vec<(f64, Vec<f64>)>) -> Layer {
    Layer { neurons }
  }

  pub fn process(&self, inputs: &Vec<f64>) -> Vec<f64> {
    self.neurons.iter()
      .map(|neuron| {
        let (bias, weights) = neuron;

        product(&weights, inputs) + bias
      })
      .map(sigmoid)
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

#[allow(dead_code)]
fn ReLU(x: f64) -> f64 {
  0_f64.max(x)
}

#[allow(dead_code)]
fn leaky_ReLU(x: f64) -> f64 {
  (0.01_f64 * x).max(x)
}

fn sigmoid(x: f64) -> f64 {
  1_f64 / (1_f64 + std::f64::consts::E.powf(-x))
}
