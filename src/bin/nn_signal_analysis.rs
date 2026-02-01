//! Neural Network Signal Analysis
//!
//! Analyzes NN output signals under different conditions:
//! - Various mutation counts (10, 20, 40, 60)
//! - Different numbers of triggered input neurons (0, 1-5, 10, 15)
//!
//! Goal: Understand what output signal strengths result from various input configurations
//! to inform activation function tuning.

use predator_vs_prey::brain_neural_network::{act_angle, act_speed, sigmoid, NeuralNetwork};
use predator_vs_prey::settings::{bias, PRED_SIGHT_COUNT, PREY_SIGHT_COUNT};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs::File;
use std::io::Write;

/// Configuration for a test scenario
#[derive(Clone)]
struct TestConfig {
    mutation_count: usize,
    triggered_count: usize,
}

/// Statistics for a batch of test runs
#[derive(Clone, Default)]
struct OutputStats {
    // Current settings
    speed_mean: f32,
    speed_std: f32,
    angle_mean: f32,
    angle_std: f32,

    // RAW inputs to activation function
    pre_speed_mean: f32,
    pre_speed_std: f32,
    pre_speed_min: f32,
    pre_speed_max: f32,

    pre_angle_mean: f32,
    pre_angle_std: f32,
    pre_angle_min: f32,
    pre_angle_max: f32,

    // Metadata
    num_networks: usize,
    samples_per_network: usize,
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║             Neural Network Signal Analysis (Signal = 1.0)                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝\n");

    // First: analyze the activation functions themselves
    analyze_activation_functions();

    // Configuration
    let mutation_counts = vec![10, 20, 40, 60, 100];
    let triggered_inputs_counts = vec![0, 1, 2, 5, 10, 15, 20];

    let networks_per_config = 50; // How many distinct NNs to create
    let samples_per_network = 20; // How many random input patterns to test per NN

    let num_inputs = PREY_SIGHT_COUNT;

    println!("Test Parameters:");
    println!("  Input neurons: {}", num_inputs);
    println!("  Mutation counts: {:?}", mutation_counts);
    println!("  Triggered inputs: {:?}", triggered_inputs_counts);
    println!("  Networks per config: {}", networks_per_config);
    println!("  Samples per network: {}", samples_per_network);
    println!(
        "  Total samples per config: {}",
        networks_per_config * samples_per_network
    );
    println!("  Bias value: {}", bias());
    println!();

    let mut all_results: Vec<(TestConfig, OutputStats)> = Vec::new();

    // Run tests
    for &mutations in &mutation_counts {
        println!("Testing {} mutations...", mutations);

        for &triggered in &triggered_inputs_counts {
            // Only test valid triggered counts
            if triggered > num_inputs {
                continue;
            }

            let config = TestConfig {
                mutation_count: mutations,
                triggered_count: triggered,
            };

            let stats = run_nested_batch_test(
                &config,
                num_inputs,
                networks_per_config,
                samples_per_network,
            );

            // Print concise summary for this config
            println!(
                "  Triggered: {:2} -> Pre-Speed: {:+.3}±{:.3}  Pre-Angle: {:+.3}±{:.3}",
                triggered,
                stats.pre_speed_mean,
                stats.pre_speed_std,
                stats.pre_angle_mean,
                stats.pre_angle_std
            );

            all_results.push((config, stats));
        }
        println!();
    }

    // Additional test: energy factor impact
    println!("\n--- Energy Factor Impact Analysis ---");
    analyze_energy_factor_impact(num_inputs, 40);

    // Write results to files
    write_csv("nn_signal_analysis.csv", &all_results);
    write_markdown_report("nn_signal_analysis.md", &all_results, num_inputs);

    println!("\n✓ Results written to:");
    println!("  - nn_signal_analysis.csv");
    println!("  - nn_signal_analysis.md");
}

/// Analyze the activation functions in isolation
fn analyze_activation_functions() {
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│                    Activation Function Analysis                             │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘\n");

    println!("act_speed (sigmoid params: a=4.0, b=3.0):");
    println!("  Input -> Output");
    for x in [-1.0, 0.0, 0.5, 0.75, 1.0, 2.0].iter() {
        println!("  {:>5.2} -> {:.4}", x, act_speed(*x));
    }
    println!();

    println!("act_angle (tanh) function:");
    println!("  Input -> act_angle(x)");
    println!("  ────────────────────────");
    for x in [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0].iter() {
        println!("  {:>6.2} -> {:>+6.4}", x, act_angle(*x));
    }
    println!();

    // Key thresholds
    println!("Key thresholds for act_speed:");
    println!(
        "  - act_speed(0.0) = {:.4} <- ZERO input output",
        act_speed(0.0)
    );
    println!(
        "  - To get speed > 0.05: need x > {:.2}",
        find_threshold(0.05)
    );
    println!(
        "  - To get speed > 0.50: need x > {:.2}",
        find_threshold(0.50)
    );
    println!(
        "  - To get speed > 0.90: need x > {:.2}",
        find_threshold(0.90)
    );
    println!();
}

/// Binary search to find input value that produces target output
fn find_threshold(target: f32) -> f32 {
    let mut low = -10.0;
    let mut high = 10.0;
    for _ in 0..100 {
        let mid = (low + high) / 2.0;
        if act_speed(mid) < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    (low + high) / 2.0
}

/// Run a nested test: Many NNs, each tested with many input patterns
fn run_nested_batch_test(
    config: &TestConfig,
    num_inputs: usize,
    num_networks: usize,
    samples_per_network: usize,
) -> OutputStats {
    let mut all_speeds = Vec::with_capacity(num_networks * samples_per_network);
    let mut all_angles = Vec::with_capacity(num_networks * samples_per_network);
    let mut all_pre_speeds = Vec::with_capacity(num_networks * samples_per_network);
    let mut all_pre_angles = Vec::with_capacity(num_networks * samples_per_network);

    for net_idx in 0..num_networks {
        let mut rng =
            StdRng::seed_from_u64((net_idx as u64) + (config.mutation_count as u64) * 1000);

        // Create 1 NN
        let mut nn = NeuralNetwork::new(num_inputs, 2, config.mutation_count, bias(), &mut rng);

        // Test it with multiple random input patterns
        for sample_idx in 0..samples_per_network {
            // Seed for inputs depends on net AND sample, so we get unique patterns
            let input_seed = (net_idx * 1000 + sample_idx) as usize;

            let inputs = create_input_vector(
                num_inputs,
                config.triggered_count,
                1.0, // Always signal 1.0 as requested
                input_seed,
            );

            let (outputs, pre_act) = nn.forward_debug(&inputs, 1.0);

            all_speeds.push(outputs[0]);
            all_angles.push(outputs[1]);
            all_pre_speeds.push(pre_act[0]);
            all_pre_angles.push(pre_act[1]);
        }
    }

    compute_stats(
        &all_speeds,
        &all_angles,
        &all_pre_speeds,
        &all_pre_angles,
        num_networks,
        samples_per_network,
    )
}

/// Create an input vector with specific number of inputs set to 1.0
fn create_input_vector(total: usize, triggered: usize, signal_value: f32, seed: usize) -> Vec<f32> {
    let mut inputs = vec![0.0; total];

    // Deterministically select which inputs to trigger based on seed
    let mut rng = StdRng::seed_from_u64(seed as u64 + 9999);
    use rand::seq::SliceRandom;
    let mut indices: Vec<usize> = (0..total).collect();
    indices.shuffle(&mut rng);

    for &idx in indices.iter().take(triggered) {
        inputs[idx] = signal_value;
    }

    inputs
}

/// Compute statistics from output vectors
fn compute_stats(
    speeds: &[f32],
    angles: &[f32],
    pre_speeds: &[f32],
    pre_angles: &[f32],
    num_nets: usize,
    samples_per_net: usize,
) -> OutputStats {
    let n = speeds.len() as f32;

    let calc_mean_std = |data: &[f32]| {
        let mean = data.iter().sum::<f32>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        (
            mean,
            variance.sqrt(),
            data.iter().cloned().fold(f32::INFINITY, f32::min),
            data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        )
    };

    let (speed_mean, speed_std, _, _) = calc_mean_std(speeds);
    let (angle_mean, angle_std, _, _) = calc_mean_std(angles);
    let (pre_speed_mean, pre_speed_std, pre_speed_min, pre_speed_max) = calc_mean_std(pre_speeds);
    let (pre_angle_mean, pre_angle_std, pre_angle_min, pre_angle_max) = calc_mean_std(pre_angles);

    OutputStats {
        speed_mean,
        speed_std,
        angle_mean,
        angle_std,
        pre_speed_mean,
        pre_speed_std,
        pre_speed_min,
        pre_speed_max,
        pre_angle_mean,
        pre_angle_std,
        pre_angle_min,
        pre_angle_max,
        num_networks: num_nets,
        samples_per_network: samples_per_net,
    }
}

/// Analyze how energy factor affects outputs
fn analyze_energy_factor_impact(num_inputs: usize, mutations: usize) {
    let mut rng = StdRng::seed_from_u64(12345);
    let mut nn = NeuralNetwork::new(num_inputs, 2, mutations, bias(), &mut rng);

    // Test with 5 triggered inputs at medium signal
    let inputs = create_input_vector(num_inputs, 5, 1.0, 0);

    println!("Energy Factor -> [Speed, Angle] (with 5 triggered inputs, signal=1.0)");
    println!("─────────────────────────────────────────────────────────────────────");

    for energy_pct in [0.1, 0.25, 0.5, 0.75, 1.0].iter() {
        let (outputs, _) = nn.forward_debug(&inputs, *energy_pct);
        println!(
            "  Energy {:>3.0}% -> Speed: {:.4}, Angle: {:+.4}",
            energy_pct * 100.0,
            outputs[0],
            outputs[1]
        );
    }
}

/// Write results to CSV file
fn write_csv(path: &str, results: &[(TestConfig, OutputStats)]) {
    let mut file = File::create(path).expect("Failed to create CSV file");

    writeln!(
        file,
        "mutations,triggered_inputs,pre_speed_mean,pre_speed_std,pre_speed_min,pre_speed_max,pre_angle_mean,pre_angle_std,pre_angle_min,pre_angle_max,post_speed_mean,post_speed_std,post_angle_mean,post_angle_std"
    )
    .unwrap();

    for (config, stats) in results {
        writeln!(
            file,
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            config.mutation_count,
            config.triggered_count,
            stats.pre_speed_mean,
            stats.pre_speed_std,
            stats.pre_speed_min,
            stats.pre_speed_max,
            stats.pre_angle_mean,
            stats.pre_angle_std,
            stats.pre_angle_min,
            stats.pre_angle_max,
            stats.speed_mean,
            stats.speed_std,
            stats.angle_mean,
            stats.angle_std
        )
        .unwrap();
    }
}

/// Write markdown report
fn write_markdown_report(path: &str, results: &[(TestConfig, OutputStats)], num_inputs: usize) {
    let mut file = File::create(path).expect("Failed to create markdown file");

    writeln!(file, "# Neural Network Signal Analysis (Signal=1.0)\n").unwrap();
    writeln!(
        file,
        "**Parameters**: Inputs={}, Networks={}, Samples/Net={}\n",
        num_inputs, results[0].1.num_networks, results[0].1.samples_per_network
    )
    .unwrap();

    // Timestamp
    use std::process::Command;
    let timestamp = Command::new("date")
        .arg("+%Y-%m-%d %H:%M:%S")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "Unknown".to_string());
    writeln!(file, "**Generated**: {}\n", timestamp).unwrap();

    // Tables by mutation count
    for mutations in [10, 20, 40, 60, 100] {
        writeln!(file, "### {} Mutations\n", mutations).unwrap();
        writeln!(
            file,
            "| Triggered | Pre-Speed Mean ± Std | Pre-Speed Range | Pre-Angle Mean ± Std | Pre-Angle Range |"
        )
        .unwrap();
        writeln!(
            file,
            "|----------:|---------------------:|----------------:|---------------------:|----------------:|"
        )
        .unwrap();

        for (config, stats) in results
            .iter()
            .filter(|(c, _)| c.mutation_count == mutations)
        {
            writeln!(
                file,
                "| {} | **{:.3}** ± {:.3} | [{:.2}, {:.2}] | **{:.3}** ± {:.3} | [{:.2}, {:.2}] |",
                config.triggered_count,
                stats.pre_speed_mean,
                stats.pre_speed_std,
                stats.pre_speed_min,
                stats.pre_speed_max,
                stats.pre_angle_mean,
                stats.pre_angle_std,
                stats.pre_angle_min,
                stats.pre_angle_max
            )
            .unwrap();
        }
        writeln!(file, "").unwrap();
    }

    // Summary
    writeln!(file, "## Summary Analysis\n").unwrap();
    writeln!(file, "Comparison of signal strength scaling (Triggered=5 vs 20) at highest mutation count (100):\n").unwrap();

    if let Some((_, low)) = results
        .iter()
        .find(|(c, _)| c.mutation_count == 100 && c.triggered_count == 5)
    {
        if let Some((_, high)) = results
            .iter()
            .find(|(c, _)| c.mutation_count == 100 && c.triggered_count == 20)
        {
            writeln!(
                file,
                "- **Speed Signal**: 5 inputs -> {:.4}, 20 inputs -> {:.4}",
                low.pre_speed_mean, high.pre_speed_mean
            )
            .unwrap();
            writeln!(
                file,
                "- **Angle Signal**: 5 inputs -> {:.4}, 20 inputs -> {:.4}",
                low.pre_angle_mean, high.pre_angle_mean
            )
            .unwrap();
        }
    }
}
