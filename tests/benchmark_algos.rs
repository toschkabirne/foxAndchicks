use predator_vs_prey::brain_neural_network::NeuralNetwork;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::time::Instant;

#[test]
fn benchmark_sorting_algos() {
    let mut rng = StdRng::seed_from_u64(42);

    // Setup a reasonably large network
    let num_inputs = 10;
    let num_outputs = 2;
    let initial_mutations = 0; // We will manually build up to avoid chaos
    let bias = 1.0;

    let mut nn = NeuralNetwork::new(num_inputs, num_outputs, initial_mutations, bias, &mut rng);

    // Add many hidden neurons
    let num_hidden = 300; // Large enough to make O(N^3) visible
    for _ in 0..num_hidden {
        nn.add_neuron();
    }

    // Randomly add connections to make it dense but acyclic (DAG)
    let in_bi = nn.num_inputs + 1;
    let hidden_start = in_bi + nn.num_outputs;

    println!("Building network with {} hidden neurons...", num_hidden);

    // Add N random valid connections
    let num_connections = 1000;
    for _ in 0..num_connections {
        let source_candidates: Vec<usize> = (hidden_start..nn.neuron_number).collect();
        let target_candidates: Vec<usize> = (in_bi..nn.neuron_number).collect();

        if source_candidates.is_empty() || target_candidates.is_empty() {
            continue;
        }

        let s = source_candidates[rng.gen_range(0..source_candidates.len())];
        let t = target_candidates[rng.gen_range(0..target_candidates.len())];

        // We use the existing safe `create_new_connection` logic which calls infinity_loop internally
        // or just manually try adding.
        if !nn.infinity_loop(s, t) {
            nn.add_connection(s, t, Some(0.5), &mut rng);
        }
    }

    println!(
        "Network built. Nodes: {}, Connections established.",
        nn.neuron_number
    );

    // Benchmarking
    let iterations = 100;

    let mut total_time_infinity = std::time::Duration::new(0, 0);
    let mut total_time_kahn = std::time::Duration::new(0, 0);

    let mut matches = 0;

    println!("Running {} iterations...", iterations);

    for _ in 0..iterations {
        // Pick random hidden source and target
        let s = rng.gen_range(hidden_start..nn.neuron_number);
        let t = rng.gen_range(hidden_start..nn.neuron_number);

        // Clone for independent testing since they modify state (dummy edge)
        let mut nn_kahn = nn.clone();

        // Measure Infinity Loop
        // infinity_loop modifies nn state (temp weight, eval_order).
        // We clone to ensure no interference if one side effects differently.
        let mut nn_inf = nn.clone();

        let start_inf = Instant::now();
        let res_inf = nn_inf.infinity_loop(s, t);
        total_time_infinity += start_inf.elapsed();

        // Measure Kahn
        let start_kahn = Instant::now();
        let res_kahn = nn_kahn.kahn_algorithm(s, t);
        total_time_kahn += start_kahn.elapsed();

        if res_inf == res_kahn {
            matches += 1;
        } else {
            println!(
                "MISMATCH! source: {}, target: {}, inf: {}, kahn: {}",
                s, t, res_inf, res_kahn
            );
        }
    }

    println!("--------------------------------------------------");
    println!("Results over {} iterations:", iterations);
    println!("Infinity Loop Total Time: {:?}", total_time_infinity);
    println!("Kahn Algo Total Time:     {:?}", total_time_kahn);
    println!("Agreement: {}/{}", matches, iterations);
    if total_time_kahn.as_secs_f64() > 0.0 {
        println!(
            "Speedup: {:.2}x",
            total_time_infinity.as_secs_f64() / total_time_kahn.as_secs_f64()
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
