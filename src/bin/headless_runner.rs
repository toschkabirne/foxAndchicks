use predator_vs_prey::game::Game;
use predator_vs_prey::settings;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
// use std::collections::HashMap;

use std::env;

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Serialize)]
struct SimResult {
    survived: bool,
    steps: i64,
    avg_step_time: f64,
    final_predators: usize,
    final_preys: usize,
    parameters: Value,

    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn main() {
    // Python argparse equivalent:
    // --params JSON_STRING (required)
    // --max_steps int (default 5000)
    let params_str = match parse_arg("--params") {
        Some(s) => s,
        None => {
            // match python behavior: print {"error": "..."}
            println!(
                "{}",
                json!({ "error": "Missing required --params" }).to_string()
            );
            return;
        }
    };

    let max_steps: i64 = parse_arg("--max_steps")
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(5000);

    let params_json: Value = match serde_json::from_str(&params_str) {
        Ok(v) => v,
        Err(e) => {
            println!(
                "{}",
                json!({ "error": format!("Invalid params JSON: {e}") }).to_string()
            );
            return;
        }
    };

    // Run
    let result = run_simulation(&params_json, max_steps);

    // Print JSON result on stdout (like Python)
    match serde_json::to_string(&result) {
        Ok(s) => println!("{s}"),
        Err(e) => println!(
            "{}",
            json!({ "error": format!("Failed to serialize result: {e}") }).to_string()
        ),
    }
}
fn parse_arg(flag: &str) -> Option<String> {
    let args: Vec<String> = env::args().collect();
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
        i += 1;
    }
    None
}

fn run_simulation(params_raw: &serde_json::Value, max_steps: i64) -> SimResult {
    let params: Params = match serde_json::from_value(params_raw.clone()) {
        Ok(p) => p,
        Err(e) => {
            return SimResult {
                survived: false,
                steps: 0,
                avg_step_time: 0.0,
                final_predators: 0,
                final_preys: 0,
                parameters: params_raw.clone(),
                error: Some(format!("Failed to parse params: {e}")),
            }
        }
    };

    // Apply params (global patch approach)
    // Ideally do this with a guard that restores after the run.
    if let Some(v) = params.add_neuron {
        settings::set_add_neuron(v);
    }
    if let Some(v) = params.add_weight {
        settings::set_add_weight(v);
    }
    if let Some(v) = params.change_weight {
        settings::set_change_weight(v);
    }
    if let Some(v) = params.pred_init_mut {
        settings::set_pred_init_mut(v);
    }
    if let Some(v) = params.prey_init_mut {
        settings::set_prey_init_mut(v);
    }

    // Create game using your encapsulated logic
    let mut game = Game::new(
        None, // DataManager is not used in headless mode
        params.max_pred_count.unwrap(),
        params.max_prey_count.unwrap(),
        params.pred_init_numb.unwrap(),
        params.prey_init_numb.unwrap(),
    );

    let start = Instant::now();
    let mut step: i64 = 0;

    while step < max_steps {
        if game.predator_count() == 0 || game.prey_count() == 0 {
            break;
        }
        let _ = game.next_frame(); // ignore Frame in headless
        step += 1;
    }

    let duration = start.elapsed().as_secs_f64();
    let avg_step_time = if step > 0 {
        duration / step as f64
    } else {
        0.0
    };
    let survived = game.predator_count() > 0 && game.prey_count() > 0;

    SimResult {
        survived,
        steps: step,
        avg_step_time,
        final_predators: game.predator_count(),
        final_preys: game.prey_count(),
        parameters: params_raw.clone(),
        error: None,
    }
}
