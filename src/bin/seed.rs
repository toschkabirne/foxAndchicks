use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..10 {
        println!("{}", rng.gen_range(-1.0..1.0));
    }
}
