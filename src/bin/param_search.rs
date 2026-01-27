use ::rand::rngs::StdRng;
use ::rand::{Rng, SeedableRng};
use colored::*;
use predator_vs_prey::settings;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::process::Command;
// use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Params {
    #[serde(rename = "ADD_NEURON")]
    add_neuron: Option<f32>,
    #[serde(rename = "ADD_WEIGHT")]
    add_weight: Option<f32>,
    #[serde(rename = "CHANGE_WEIGHT")]
    change_weight: Option<f32>,
    #[serde(rename = "PRED_INIT_MUT")]
    pred_init_mut: Option<usize>,
    #[serde(rename = "PREY_INIT_MUT")]
    prey_init_mut: Option<usize>,
    #[serde(rename = "MAX_PRED_COUNT")]
    max_pred_count: Option<usize>,
    #[serde(rename = "MAX_PREY_COUNT")]
    max_prey_count: Option<usize>,
    #[serde(rename = "PRED_INIT_NUMB")]
    pred_init_numb: Option<usize>,
    #[serde(rename = "PREY_INIT_NUMB")]
    prey_init_numb: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SimResult {
    survived: bool,
    steps: i64,
    avg_step_time: f64,
    final_predators: usize,
    final_preys: usize,
    parameters: Value,
    error: Option<String>,
}

fn generate_random_params() -> Params {
    let mut rng = StdRng::from_seed(settings::SEED);
    Params {
        add_neuron: Some(rng.gen_range(0.01..0.9)),
        add_weight: Some(rng.gen_range(0.1..0.9)),
        change_weight: Some(rng.gen_range(0.1..0.9)),
        pred_init_mut: Some(rng.gen_range(10..60)),
        prey_init_mut: Some(rng.gen_range(5..30)),
        max_pred_count: Some(rng.gen_range(50..500)),
        max_prey_count: Some(rng.gen_range(100..2000)),
        pred_init_numb: Some(rng.gen_range(10..150)),
        prey_init_numb: Some(rng.gen_range(20..500)),
    }
}

fn run_trial(params: &Params) -> Option<SimResult> {
    let params_json = serde_json::to_string(params).ok()?;

    // Assume executable is in target/release/headless_runner
    // We use "cargo run --bin headless_runner --release -- --params ..." for simplicity if direct binary path is tricky,
    // but calling binary directly is faster.
    // Let's rely on finding the binary relative to current dir.
    let output = Command::new("target/release/headless_runner")
        .arg("--params")
        .arg(&params_json)
        .arg("--max_steps")
        .arg("20000") // Longer max steps for search
        .output();

    // Fallback to cargo run if binary not found (slower/noisy)
    let output = match output {
        Ok(o) => o,
        Err(_) => Command::new("cargo")
            .arg("run")
            .arg("--quiet")
            .arg("--release")
            .arg("--bin")
            .arg("headless_runner")
            .arg("--")
            .arg("--params")
            .arg(&params_json)
            .arg("--max_steps")
            .arg("2000")
            .output()
            .ok()?,
    };

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).ok()
}

fn main() {
    let num_trials = 50; // run 50 parallel sims
    println!("Running {} parameter search trials...", num_trials);

    // Ensure we have a build first (optional but good)
    let _ = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("--bin")
        .arg("headless_runner")
        .status();

    let results: Vec<SimResult> = (0..num_trials)
        .into_par_iter()
        .map(|_i| {
            let params = generate_random_params();
            if let Some(res) = run_trial(&params) {
                print!(".");
                res
            } else {
                print!("x");
                SimResult {
                    survived: false,
                    steps: 0,
                    avg_step_time: 0.0,
                    final_predators: 0,
                    final_preys: 0,
                    parameters: json!(params),
                    error: Some("Run failed".to_string()),
                }
            }
        })
        .collect();

    println!("\n\nSearch complete. Analyzing results...");

    // Filter for survived
    let survived: Vec<&SimResult> = results.iter().filter(|r| r.survived).collect();
    println!("Survived: {}/{}", survived.len(), num_trials);

    if let Some(best) = survived.iter().max_by_key(|r| r.steps) {
        println!("\nBest Result (Longest Survival):");
        println!("{}", serde_json::to_string_pretty(best).unwrap().green());
    } else {
        println!("\nNo runs survived to max_steps.");
        if let Some(longest) = results.iter().max_by_key(|r| r.steps) {
            println!("Longest attempt: {} steps", longest.steps);
            println!("Params: {}", longest.parameters);
        }
    }
}
