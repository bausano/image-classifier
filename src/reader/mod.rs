pub mod digit;

use self::digit::Digit;

pub fn read_digits() -> Vec<Digit> {
  let mut digits: Vec<Digit> = Vec::new();

  let lines = include_str!("../../data/input.txt").split("\n");

  for line in lines {
    if line.len() == 0 {
      continue;
    }

    let mut grid: Vec<u8> = Vec::new();

    for number in line.split(",") {
      grid.push(
        number.parse::<u8>().unwrap()
      );
    }

    let class = grid.pop().unwrap();

    digits.push(Digit::new(class, grid));
  }

  digits
}
