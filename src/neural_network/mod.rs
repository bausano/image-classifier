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
          0_f64,
          (0..(layer_schema[i] - 1))
            .map(|_| rng.gen::<f64>() * 2_f64 - 1_f64)
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
    let activation_matrix = self.calculate_activation_matrix(&training_matrix);

    for digit in activation_matrix.iter() {
      println!("-------");

      for col in digit.iter() {
        for cell in col.iter() {
          print!("{:.2}, ", cell);
        }

        println!("\n");
      }
    }

    //let mut cost = 0_f64;

    let iterator = activation_matrix.iter().zip(training_matrix.iter());

    iterator.fold(0_f64, |carry, (xd, (target, _i))| {
      let outputs = xd.last().unwrap();
      let target = usize::from(target.clone());

      carry + outputs.iter().enumerate()
        .fold(0_f64, |sum, (i, o)| {
          let res = if i == target { 1_f64 } else { 0_f64 };

          sum + (o - res).powi(2)
        })
    })

    /*for i in 0..training_matrix.len() {
      let target = training_matrix[i].0;

      let outputs = activation_matrix[i].last().unwrap();

      cost += outputs.iter().enumerate()
        .fold(0_f64, |sum, (i, o)| {
          let res = if i == usize::from(target) { 1_f64 } else { 0_f64 };

          sum + (o - res).powi(2)
        });
    }

    cost*/
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
