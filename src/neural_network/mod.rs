mod layer;

use self::layer::Layer;

pub struct Network {
  layers: Vec<Layer>
}

impl Network {
  pub fn from(schema: Vec<Vec<(f64, Vec<f64>)>>) -> Network {
    Network {
      layers: schema.into_iter()
        .map(|neurons| Layer { neurons })
        .collect()
    }
  }

  pub fn train(&self, data: Vec<f64>, expected: u8) {
    let classes: Vec<f64> = self.compute(data);

    //
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
      .fold(inputs, |signal, layer| layer.process(signal))
  }
}
