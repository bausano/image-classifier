pub mod digit;

use self::digit::Digit;

pub fn read_digits() -> Vec<Digit> {
  include_str!("../../data/input.txt").split("\n")
    .filter_map(|line| {
      if line.len() == 0 {
        return None;
      }

      let mut numbers: Vec<&str> = line.split(",").collect();

      let class: u8 = numbers.pop().unwrap().parse::<u8>().unwrap();

      let grid: Vec<f64> = numbers.iter()
        .map(|x| x.parse::<f64>().unwrap() / 16_f64)
        .collect();

      Some(Digit::new(class, grid))
    })
    .collect()
}
