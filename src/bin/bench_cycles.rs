use predator_vs_prey::brain_neural_network::NeuralNetwork;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const NUM_INPUTS: usize = 10;
const NUM_OUTPUTS: usize = 2;
const BIAS: f32 = 1.0;

const QUERIES: usize = 10_000;
const WARMUP_ITERS: usize = 500;

fn median_duration(mut xs: Vec<Duration>) -> Duration {
    xs.sort();
    xs[xs.len() / 2]
}

fn build_network(hidden_neurons: usize, extra_connections: usize, seed: u64) -> NeuralNetwork {
    let mut rng = StdRng::seed_from_u64(seed);

    let initial_mutations = 0;
    let mut nn = NeuralNetwork::new(NUM_INPUTS, NUM_OUTPUTS, initial_mutations, BIAS, &mut rng);

    for _ in 0..hidden_neurons {
        nn.add_neuron();
    }

    let in_bi = nn.num_inputs + 1;
    let hidden_start = in_bi + nn.num_outputs;

    let mut attempts = 0usize;
    let max_attempts = extra_connections.saturating_mul(20).max(extra_connections);

    let mut added = 0usize;
    while added < extra_connections && attempts < max_attempts {
        attempts += 1;

        if hidden_start >= nn.neuron_number || in_bi >= nn.neuron_number {
            break;
        }

        let s = rng.gen_range(hidden_start..nn.neuron_number);
        let t = rng.gen_range(in_bi..nn.neuron_number);

        // keep DAG
        if !nn.infinity_loop(s, t) {
            nn.add_connection(s, t, Some(0.5), &mut rng);
            added += 1;
        }
    }

    nn
}

fn generate_pairs(nn: &NeuralNetwork, seed: u64, queries: usize) -> Vec<(usize, usize)> {
    let mut rng = StdRng::seed_from_u64(seed);

    let in_bi = nn.num_inputs + 1;
    let hidden_start = in_bi + nn.num_outputs;

    let mut pairs = Vec::with_capacity(queries);
    for _ in 0..queries {
        let s = rng.gen_range(hidden_start..nn.neuron_number);
        let t = rng.gen_range(hidden_start..nn.neuron_number);
        pairs.push((s, t));
    }
    pairs
}

fn warmup(
    nn_inf: &mut NeuralNetwork,
    nn_kahn: &mut NeuralNetwork,
    pairs: &[(usize, usize)],
) -> u64 {
    let mut sink: u64 = 0;
    for &(s, t) in pairs.iter().take(WARMUP_ITERS.min(pairs.len())) {
        sink += nn_inf.infinity_loop(s, t) as u64;
        sink += nn_kahn.kahn_algorithm(s, t) as u64;
    }
    sink
}

fn measure_infinity(nn: &mut NeuralNetwork, pairs: &[(usize, usize)]) -> (Duration, u64, Vec<u8>) {
    let mut sink: u64 = 0;
    let mut results: Vec<u8> = Vec::with_capacity(pairs.len());

    let start = Instant::now();
    for &(s, t) in pairs {
        let r = nn.infinity_loop(s, t) as u8;
        sink += r as u64;
        results.push(r);
    }
    (start.elapsed(), sink, results)
}

fn measure_kahn(
    nn: &mut NeuralNetwork,
    pairs: &[(usize, usize)],
    baseline: &[u8],
) -> (Duration, u64, usize) {
    let mut sink: u64 = 0;
    let mut mismatches: usize = 0;

    let start = Instant::now();
    for (i, &(s, t)) in pairs.iter().enumerate() {
        let r = nn.kahn_algorithm(s, t) as u8;
        sink += r as u64;
        if baseline[i] != r {
            mismatches += 1;
        }
    }
    (start.elapsed(), sink, mismatches)
}

fn main() {
    let v_list = [50usize, 100, 200, 400, 800];
    let seeds = [1u64, 2, 3];
    let edge_multipliers = [2usize, 8];

    println!("Benchmark: cycle detection (infinity_loop vs kahn_algorithm)");
    println!("Run: cargo run --release --bin bench_cycles");
    println!("Fixed queries per case: {}", QUERIES);
    println!("Report: median runtime across seeds + speedup (inf/kahn)\n");

    println!("V_hidden,E_extra,queries,median_kahn_ms,median_inf_ms,speedup_kahn_over_inf,median_mismatches");

    for &v_hidden in &v_list {
        for &mul in &edge_multipliers {
            let e_extra = mul * v_hidden;

            let mut inf_times = Vec::with_capacity(seeds.len());
            let mut kahn_times = Vec::with_capacity(seeds.len());
            let mut mismatches_list: Vec<usize> = Vec::with_capacity(seeds.len());

            for &seed in &seeds {
                let nn_base = build_network(v_hidden, e_extra, seed);

                // deterministic identical pairs for both algorithms
                let pair_seed = 10_000u64
                    ^ seed.wrapping_mul(1_000_003)
                    ^ (v_hidden as u64).wrapping_mul(31)
                    ^ (e_extra as u64).wrapping_mul(131);

                let pairs = generate_pairs(&nn_base, pair_seed, QUERIES);

                // independent scratch state per algorithm, but no per-query clone
                let mut nn_inf = nn_base.clone();
                let mut nn_kahn = nn_base.clone();

                let warm_sink = warmup(&mut nn_inf, &mut nn_kahn, &pairs);
                if warm_sink == u64::MAX {
                    eprintln!("(unreachable) warm_sink hit MAX");
                }

                // measure infinity and store baseline outputs
                let (t_inf, sink_inf, baseline) = measure_infinity(&mut nn_inf, &pairs);
                let (t_kahn, sink_kahn, mismatches) = measure_kahn(&mut nn_kahn, &pairs, &baseline);

                // keep sinks “alive”
                if sink_inf == 123456789 && sink_kahn == 987654321 {
                    eprintln!("(unreachable) sinks matched magic values");
                }

                inf_times.push(t_inf);
                kahn_times.push(t_kahn);
                mismatches_list.push(mismatches);
            }

            let med_inf = median_duration(inf_times);
            let med_kahn = median_duration(kahn_times);

            mismatches_list.sort();
            let med_mismatches = mismatches_list[mismatches_list.len() / 2];

            let med_inf_ms = med_inf.as_secs_f64() * 1000.0;
            let med_kahn_ms = med_kahn.as_secs_f64() * 1000.0;

            let speedup = if med_inf.as_secs_f64() > 0.0 {
                med_kahn.as_secs_f64() / med_inf.as_secs_f64()
            } else {
                f64::INFINITY
            };

            println!(
                "{},{},{},{:.3},{:.3},{:.3},{}",
                v_hidden, e_extra, QUERIES, med_kahn_ms, med_inf_ms, speedup, med_mismatches
            );
        }
    }
}
