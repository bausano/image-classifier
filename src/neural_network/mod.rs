mod layer;

use self::layer::Layer;
use rand::prelude::*;

pub struct Network {
  pub layers: Vec<Layer>
}

impl Network {
  pub fn new(layer_schema: Vec<u8>) -> Network {
    let mut rng = rand::thread_rng();
    let mut layers: Vec<Vec<(f64, Vec<f64>)>> = Vec::new();

    for i in 1..layer_schema.len() {
      let mut layer: Vec<(f64, Vec<f64>)> = Vec::new();

      for _ in 0..layer_schema[i] {
        layer.push((
          rng.gen::<f64>(),
          (0..(layer_schema[i] - 1))
            .map(|_| rng.gen::<f64>())
            .collect()
        ));
      }

      layers.push(layer);
    }

    Network::from(layers)
  }

  pub fn from(schema: Vec<Vec<(f64, Vec<f64>)>>) -> Network {
    Network {
      layers: schema.into_iter()
        .map(|neurons| Layer::from(neurons))
        .collect()
    }
  }

  pub fn classify(&self, inputs: Vec<f64>) -> u8 {
    let mut i: u8 = 0;
    let mut res: Option<u8> = Some(i);
    let mut max: Option<f64> = None;

    for &class in self.compute(inputs).iter() {
      match max {
        None => max = Some(class),
        Some(x) => if x < class {
          max = Some(class);
          res = Some(i);
        }
      }

      i += 1;
    }

    // We use option here only to make the first res assigment
    // more comportable, hence can safely unwrap.
    res.unwrap()
  }

  fn compute(&self, inputs: Vec<f64>) -> Vec<f64> {
    self.layers.iter()
      .fold(inputs, |signal, layer| layer.process(&signal))
  }

  #[allow(dead_code)]
  pub fn train(&mut self, training_matrix: Vec<(u8, Vec<f64>)>) -> f64 {
    let mut activation_matrix = self.calculate_activation_matrix(&training_matrix);

    for i in 0..training_matrix.len() {
      let target = training_matrix[i].0;
    }

    0_f64
  }

  fn calculate_activation_matrix(&self, matrix: &Vec<(u8, Vec<f64>)>)
    -> Vec<Vec<Vec<f64>>> {
    matrix.iter()
      .map(|(_class, inputs)| self.layers.iter().fold(
        vec!(inputs.clone()),
        |mut activations, layer| {
          let output = layer.process(&activations.last().unwrap());

          activations.push(output);

          activations
      }))
      .collect()
  }
}
