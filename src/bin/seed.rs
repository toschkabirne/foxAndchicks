use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..10 {
        println!("{}", rng.next_u64());
    }
}
