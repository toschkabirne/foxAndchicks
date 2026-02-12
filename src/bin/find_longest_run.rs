use colored::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process::Command;

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

fn run_simulation_long(params_json: &Value) -> Option<SimResult> {
    let params_str = serde_json::to_string(params_json).ok()?;

    // Assume executable is in target/release/headless_runner
    // We try to reuse the built binary for efficiency
    let output = Command::new("target/release/headless_runner")
        .arg("--params")
        .arg(&params_str)
        .arg("--max_steps")
        .arg("200000")
        .output();

    let output = match output {
        Ok(o) => o,
        Err(_) => {
            // Fallback to cargo run (slower startup, but safer)
            Command::new("cargo")
                .arg("run")
                .arg("--quiet")
                .arg("--release")
                .arg("--bin")
                .arg("headless_runner")
                .arg("--")
                .arg("--params")
                .arg(&params_str)
                .arg("--max_steps")
                .arg("200000")
                .output()
                .ok()?
        }
    };

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).ok()
}

fn main() {
    let input_file = "parm_search.out";
    let file = match File::open(input_file) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open {}: {}", input_file, e);
            return;
        }
    };

    let reader = BufReader::new(file);
    let mut candidates = Vec::new();

    println!("Reading {}...", input_file);

    for line in reader.lines() {
        let line = line.unwrap_or_default();
        if line.trim().is_empty() {
            continue;
        }

        // Try to parse as SimResult. If it fails, it's likely a header line.
        if let Ok(prev_result) = serde_json::from_str::<SimResult>(&line) {
            candidates.push(prev_result);
        }
    }

    if candidates.is_empty() {
        println!("No candidate parameters found in {}.", input_file);
        return;
    }

    println!(
        "Found {} candidates. Starting long verification (200,000 frames)...",
        candidates.len()
    );

    let mut best_result: Option<SimResult> = None;

    for (i, candidate) in candidates.iter().enumerate() {
        print!("Run {}/{}: ... ", i + 1, candidates.len());
        use std::io::Write;
        std::io::stdout().flush().unwrap();

        if let Some(res) = run_simulation_long(&candidate.parameters) {
            if res.survived {
                println!("{}", "SURVIVED (200k)".green());
            } else {
                println!("{} ({} steps)", "DIED".red(), res.steps);
            }

            // Logic to find the "best".
            // 1. Survival is best.
            // 2. If multiple survive, maybe checks steps? (All survivors have 200k steps)
            // 3. If none survive, whoever lasted longest.

            match &best_result {
                None => best_result = Some(res),
                Some(current_best) => {
                    if res.steps > current_best.steps {
                        best_result = Some(res);
                    } else if res.steps == current_best.steps {
                        // Tie-breaker? Maybe population count?
                        // For now, first one wins or overwrite?
                        // "which parameter combination won" -> usually the one that survived.
                        // If both survived, they are equal in terms of "winning" the survival challenge.
                        // Let's keep the one with higher predator count as a tie breaker for "healthy ecosystem"?
                        if res.final_predators > current_best.final_predators {
                            best_result = Some(res);
                        }
                    }
                }
            }
        } else {
            println!("{}", "ERROR".red());
        }
    }

    println!("\n========================================");
    if let Some(winner) = best_result {
        println!("{}", "WINNER (Best Parameter Set):".green().bold());
        if winner.survived {
            println!("Result: Survived all 200,000 frames!");
        } else {
            println!(
                "Result: Did not survive, but lasted {} steps.",
                winner.steps
            );
        }
        println!(
            "Final State: {} Preds, {} Prey",
            winner.final_predators, winner.final_preys
        );
        println!("\nParameters:");
        println!(
            "{}",
            serde_json::to_string_pretty(&winner.parameters).unwrap()
        );
    } else {
        println!("No results obtained.");
    }
    println!("========================================");
}
