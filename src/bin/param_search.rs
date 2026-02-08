use colored::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::File;
use std::io::Write;
use std::process::Command;

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

fn generate_grid_params() -> Vec<Params> {
    let mut params_list = Vec::new();

    // Ranges
    // add_neuron: 0.07..0.15 (step 0.01) -> 7, 8, ... 15 / 100.0
    let add_neurons: Vec<f32> = (7..=15).map(|x| x as f32 / 100.0).collect();

    // add_weight: 0.5..0.7 (step 0.1) -> 0.5, 0.6, 0.7
    let add_weights: Vec<f32> = vec![0.5, 0.6, 0.7];

    // change_weight: 0.6..0.9 (step 0.1) -> 0.6, 0.7, 0.8, 0.9
    let change_weights: Vec<f32> = vec![0.6, 0.7, 0.8, 0.9];

    // mutation: 20, 30, 40 (for both pred and prey init mut)
    let mutations: Vec<usize> = vec![15];

    // Population configs: (PredInit, PreyInit, MaxPred, MaxPrey)
    // 1. 40, 160, 125, 500
    // 2. 100, 400, 600, 2400
    let pop_configs = vec![
        (50, 200, 125, 600),
        (100, 400, 150, 700),
        (100, 400, 200, 1000),
    ];

    for &an in &add_neurons {
        for &aw in &add_weights {
            for &cw in &change_weights {
                for &mut_rate in &mutations {
                    for &(pred_init, prey_init, max_pred, max_prey) in &pop_configs {
                        params_list.push(Params {
                            add_neuron: Some(an),
                            add_weight: Some(aw),
                            change_weight: Some(cw),
                            pred_init_mut: Some(mut_rate),
                            prey_init_mut: Some(mut_rate),
                            max_pred_count: Some(max_pred),
                            max_prey_count: Some(max_prey),
                            pred_init_numb: Some(pred_init),
                            prey_init_numb: Some(prey_init),
                        });
                    }
                }
            }
        }
    }

    params_list
}

fn run_trial(params: &Params) -> Option<SimResult> {
    let params_json = serde_json::to_string(params).ok()?;

    // Assume executable is in target/release/headless_runner
    let output = Command::new("target/release/headless_runner")
        .arg("--params")
        .arg(&params_json)
        .arg("--max_steps")
        .arg("40000")
        .output();

    // Fallback to cargo run if binary not found
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
            .arg("40000")
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
    let params_list = generate_grid_params();
    let total_trials = params_list.len();
    println!(
        "Running {} parameter search trials (Deterministic Grid)...",
        total_trials
    );

    // Ensure we have a build first
    let _ = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("--bin")
        .arg("headless_runner")
        .status();

    let results: Vec<SimResult> = params_list
        .par_iter()
        .map(|params| {
            if let Some(res) = run_trial(params) {
                let status = if res.survived {
                    "SURVIVED".green()
                } else {
                    "DIED    ".red()
                };

                println!(
                    "[{}] Steps: {:<5} | AddN:{:.2} AddW:{:.1} ChngW:{:.1} Mut:{} | Pop: {}-{}/{}-{}",
                    status,
                    res.steps,
                    params.add_neuron.unwrap_or(0.0),
                    params.add_weight.unwrap_or(0.0),
                    params.change_weight.unwrap_or(0.0),
                    params.pred_init_mut.unwrap_or(0),
                    params.pred_init_numb.unwrap_or(0),
                    params.max_pred_count.unwrap_or(0),
                    params.prey_init_numb.unwrap_or(0),
                    params.max_prey_count.unwrap_or(0),
                );
                res
            } else {
                println!("{}", "ERROR: Run failed".red());
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
    println!("Survived: {}/{}", survived.len(), total_trials);

    let output_file_name = "parm_search.out";
    let mut file = File::create(output_file_name).expect("Failed to create output file");

    if survived.is_empty() {
        writeln!(file, "No runs survived.").unwrap();
    } else {
        writeln!(file, "Successful Runs ({}):", survived.len()).unwrap();
        for res in &survived {
            writeln!(file, "{}", serde_json::to_string(res).unwrap()).unwrap();
        }
    }
    println!("Results saved to {}", output_file_name);

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
