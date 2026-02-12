use colored::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};

const BIN_PATH: &str = "target/release/headless_runner";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
    #[serde(rename = "SEED")]
    seed: Option<u64>,
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

fn write_results(results_file: &str, results: &Vec<SimResult>) {
    if let Ok(json) = serde_json::to_string_pretty(results) {
        let tmp_file = format!("{}.tmp", results_file);
        let _ = fs::write(&tmp_file, json);
        let _ = fs::rename(&tmp_file, results_file);
    }
}

fn generate_grid_params() -> Vec<Params> {
    let mut params_list = Vec::new();

    // Ranges
    // add_neuron: 0.07..0.15 (step 0.01) -> 7, 8, ... 15 / 100.0
    let add_neurons: Vec<f32> = (5..=15).map(|x| x as f32 / 100.0).collect();

    // add_weight: 0.5..0.7 (step 0.1) -> 0.5, 0.6, 0.7
    let add_weights: Vec<f32> = vec![0.5, 0.6, 0.7];

    // change_weight: 0.6..0.9 (step 0.1) -> 0.6, 0.7, 0.8, 0.9
    let change_weights: Vec<f32> = vec![0.6, 0.7, 0.8];

    // mutation: 20, 30, 40 (for both pred and prey init mut)
    let mutations: Vec<usize> = vec![10, 15];

    // Three different seeds for testing
    let seeds: Vec<u64> = vec![420, 12345, 61212];

    for &an in &add_neurons {
        for &aw in &add_weights {
            for &cw in &change_weights {
                for &mut_rate in &mutations {
                    for &seed in &seeds {
                        params_list.push(Params {
                            add_neuron: Some(an),
                            add_weight: Some(aw),
                            change_weight: Some(cw),
                            pred_init_mut: Some(mut_rate),
                            prey_init_mut: Some(mut_rate),
                            pred_init_numb: Some(110),
                            prey_init_numb: Some(450),
                            max_pred_count: Some(175),
                            max_prey_count: Some(800),
                            seed: Some(seed),
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

    let output = Command::new(BIN_PATH)
        .arg("--params")
        .arg(&params_json)
        .arg("--max_steps")
        .arg("40000")
        .output()
        .ok()?; // If execution fails, just return None

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("Runner failed: {}", stderr);
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    match serde_json::from_str(&stdout) {
        Ok(result) => Some(result),
        Err(e) => {
            eprintln!("Failed to parse JSON from stdout: {}", e);
            eprintln!("Stdout was: {}", stdout);
            None
        }
    }
}

fn main() {
    let params_list = generate_grid_params();
    let total_trials = params_list.len();

    // Load previous results to identify failed configurations
    let results_file = "param_search_results.json";
    let mut failed_params: HashSet<String> = HashSet::new();
    let mut previous_results: Vec<SimResult> = Vec::new();

    if let Ok(content) = fs::read_to_string(results_file) {
        if let Ok(results) = serde_json::from_str::<Vec<SimResult>>(&content) {
            println!("Loaded {} previous results", results.len());
            for result in &results {
                if !result.survived {
                    // Use the JSON string as a key
                    failed_params.insert(result.parameters.to_string());
                }
            }
            previous_results = results;
            println!(
                "Skipping {} previously failed configurations",
                failed_params.len()
            );
        }
    }

    // Filter out previously failed configurations
    let params_to_test: Vec<&Params> = params_list
        .iter()
        .filter(|p| {
            let param_json = serde_json::to_value(p)
                .ok()
                .map(|v| v.to_string())
                .unwrap_or_default();
            !failed_params.contains(&param_json)
        })
        .collect();

    let tests_to_run = params_to_test.len();
    println!(
        "Running {} parameter search trials (Deterministic Grid)...",
        tests_to_run
    );
    println!("Total configurations: {}", total_trials);
    println!("Skipped (already failed): {}", total_trials - tests_to_run);

    // Build once
    let status = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("--bin")
        .arg("headless_runner")
        .status()
        .expect("Failed to start cargo build");

    if !status.success() {
        panic!("Release build of headless_runner failed.");
    }

    // Fail fast if binary does not exist
    if !Path::new(BIN_PATH).exists() {
        panic!(
            "Binary not found at '{}'. Build succeeded but file is missing.",
            BIN_PATH
        );
    }

    let results_file = "param_search_results.json";
    let results_shared = Arc::new(Mutex::new(previous_results));
    {
        let initial = results_shared.lock().unwrap().clone();
        write_results(results_file, &initial);
    }

    params_to_test.par_iter().for_each(|&params| {
        let res = if let Some(res) = run_trial(params) {
            let status = if res.survived { "SURVIVED".green() } else { "DIED    ".red() };
            println!(
                "[{}] Steps: {:<5} | Preds:{}/{} Preys:{}/{} | AddN:{:.2} AddW:{:.1} ChngW:{:.1} Mut:{} Seed:{}",
                status,
                res.steps,
                res.final_predators,
                params.max_pred_count.unwrap_or(0),
                res.final_preys,
                params.max_prey_count.unwrap_or(0),
                params.add_neuron.unwrap_or(0.0),
                params.add_weight.unwrap_or(0.0),
                params.change_weight.unwrap_or(0.0),
                params.pred_init_mut.unwrap_or(0),
                params.seed.unwrap_or(0)
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
        };

        let mut guard = results_shared.lock().unwrap();
        guard.push(res);
        write_results(results_file, &guard);
    });

    let results = results_shared.lock().unwrap().clone();

    // Save all results to file
    write_results(results_file, &results);
    println!("Results saved to {}", results_file);

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
