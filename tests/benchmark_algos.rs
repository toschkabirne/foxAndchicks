use predator_vs_prey::brain_neural_network::NeuralNetwork;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

fn build_network(seed: u64) -> NeuralNetwork {
    let mut rng = StdRng::seed_from_u64(seed);

    // Setup a reasonably large network
    let num_inputs = 30;
    let num_outputs = 2;
    let initial_mutations = 0; // build manually to keep structure stable
    let bias = 1.0;

    let mut nn = NeuralNetwork::new(num_inputs, num_outputs, initial_mutations, bias, &mut rng);

    // Add many hidden neurons
    let num_hidden = 50;
    for _ in 0..num_hidden {
        nn.add_neuron();
    }

    // Indices as in your original code
    let in_bi = nn.num_inputs + 1;
    let hidden_start = in_bi + nn.num_outputs;

    // Randomly add connections (attempt dense-ish DAG)
    let num_connections = 200;

    for _ in 0..num_connections {
        // Keep it cheap: avoid allocating Vec each time
        if hidden_start >= nn.neuron_number || in_bi >= nn.neuron_number {
            continue;
        }

        let s = rng.gen_range(hidden_start..nn.neuron_number);
        let t = rng.gen_range(in_bi..nn.neuron_number);

        // Only add if it won't create a cycle
        if !nn.infinity_loop(s, t) {
            nn.add_connection(s, t, Some(0.5), &mut rng);
        }
    }

    nn
}

#[test]
#[ignore] // run explicitly in --release, see notes below
fn benchmark_cycle_detection_algos_no_clone_bias() {
    // One seed to build a stable network shape
    let network_seed = 42u64;

    let nn_base = build_network(network_seed);

    let in_bi = nn_base.num_inputs + 1;
    let hidden_start = in_bi + nn_base.num_outputs;

    // Use a separate seed for (s,t) queries, so pair generation is deterministic too
    let query_seed = 1337u64;
    let mut rng = StdRng::seed_from_u64(query_seed);

    // More iterations (tweak as you like)
    let warmup_iters: usize = 2_000;
    let iterations: usize = 10_000;

    // Pre-generate identical (s,t) pairs for both algorithms
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let s = rng.gen_range(hidden_start..nn_base.neuron_number);
        let t = rng.gen_range(hidden_start..nn_base.neuron_number);
        pairs.push((s, t));
    }

    // Use two independent networks so any internal scratch state doesn't “help/hurt” the other.
    // No per-iteration clones, no allocation inside the timed loops.
    let mut nn_inf = nn_base.clone();
    let mut nn_kahn = nn_base.clone();

    // Warmup (not timed): prime caches, branch predictors, allocators, whatever humans invented to ruin measurement
    let mut warmup_sink: u64 = 0;
    for &(s, t) in pairs.iter().take(warmup_iters) {
        warmup_sink += nn_inf.infinity_loop(s, t) as u64;
        warmup_sink += nn_kahn.kahn_algorithm(s, t) as u64;
    }

    // --------- Measure Infinity Loop (batch timing) ----------
    let mut res_inf: Vec<u8> = Vec::with_capacity(iterations);
    let start_inf = Instant::now();
    let mut sink_inf: u64 = 0;
    for &(s, t) in &pairs {
        let r = nn_inf.infinity_loop(s, t);
        sink_inf += r as u64;
        res_inf.push(r as u8);
    }
    let time_inf: Duration = start_inf.elapsed();

    // --------- Measure Kahn (batch timing) ----------
    let mut res_kahn: Vec<u8> = Vec::with_capacity(iterations);
    let start_kahn = Instant::now();
    let mut sink_kahn: u64 = 0;
    for &(s, t) in &pairs {
        let r = nn_kahn.kahn_algorithm(s, t);
        sink_kahn += r as u64;
        res_kahn.push(r as u8);
    }
    let time_kahn: Duration = start_kahn.elapsed();

    // Compare correctness
    let mut matches = 0usize;
    for i in 0..iterations {
        if res_inf[i] == res_kahn[i] {
            matches += 1;
        }
    }

    println!("--------------------------------------------------");
    println!("Network seed: {}, query seed: {}", network_seed, query_seed);
    println!(
        "Warmup iters: {}, measured iters: {}",
        warmup_iters, iterations
    );
    println!("Infinity Loop Total Time: {:?}", time_inf);
    println!("Kahn Algo Total Time:     {:?}", time_kahn);
    println!("Agreement: {}/{}", matches, iterations);

    // Prevent “unused” weirdness
    println!(
        "Sinks (ignore): inf={}, kahn={}, warmup={}",
        sink_inf, sink_kahn, warmup_sink
    );

    if time_kahn.as_secs_f64() > 0.0 {
        println!(
            "Speedup (inf/kahn): {:.3}x",
            time_inf.as_secs_f64() / time_kahn.as_secs_f64()
        );
    } else {
        println!("Speedup: Infinite (Kahn too fast)");
    }
    println!("--------------------------------------------------");

    assert_eq!(
        matches, iterations,
        "Algorithms should agree on cycle detection"
    );
}
