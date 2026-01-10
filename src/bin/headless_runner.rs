use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
// use std::collections::HashMap;

use std::env;
use std::time::Instant;

// ---- Assumptions / integration notes (you said you'll handle integration): ----
// - Replace these `use` paths with your actual crate/module paths.
// - The sim logic expects you already have Rust equivalents of:
//   - settings (mutable for PRED_INIT_MUT / PREY_INIT_MUT if you want to "patch" at runtime)
//   - brain_neural_network (mutable global probs if you want to "patch" at runtime)
//   - Predator / Prey types + methods identical to your port:
//       Predator::new(x, y, rng), predator.get_inputs(&[PreyRc]), predator.move_step(&inputs)
//       Prey::new(x, y, rng), prey.get_inputs(&[PredatorRc], &mut Vec<PredatorRc>, rng) -> Option<Vec<f32>>
//       prey.move_step(&inputs), prey.reproduce(rng) -> Option<PreyRc>
//
// For full fidelity to the Python monkey-patching, you probably want interior-mut globals
// (OnceLock + Mutex/RwLock) for the mutation probabilities and init mut counts.

use predatorVsPrey::animals::{Predator, Prey};
use predatorVsPrey::settings;

// SpatialHash trait + struct from your port
use predatorVsPrey::spatial_hash::SpatialHash;

use std::cell::RefCell;
use std::rc::Rc;

type PredatorRc = Rc<RefCell<Predator>>;
type PreyRc = Rc<RefCell<Prey>>;

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

/// Re-implementation of headless_runner.py as Rust.
/// Mirrors behavior:
/// - patch params
/// - spawn predators/preys
/// - run until extinction or max_steps
/// - produce JSON result on stdout
fn run_simulation(params_raw: &Value, max_steps: i64) -> SimResult {
    // Parse params (tolerant)
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

    // ---- Apply parameters (Python monkey-patching equivalents) ----
    // In Python:
    //   bnn.ADD_NEURON/ADD_WEIGHT/CHANGE_WEIGHT
    //   settings.PRED_INIT_MUT / PREY_INIT_MUT
    //
    // In Rust, constants can't be patched, so you need mutable globals/config.
    // Here we just show hooks you can wire to your own config system.

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

    let mut rng = rand::thread_rng();

    // ---- Initialize entities (same as Python) ----
    let mut predators: Vec<PredatorRc> = (0..settings::PRED_INIT_NUMB)
        .map(|_| {
            let x = rng.gen_range(0.0..settings::SCREEN_WIDTH as f32);
            let y = rng.gen_range(0.0..settings::SCREEN_HEIGHT as f32);
            Rc::new(RefCell::new(Predator::new(x, y, &mut rng)))
        })
        .collect();

    let mut preys: Vec<PreyRc> = (0..settings::PREY_INIT_NUMB)
        .map(|_| {
            let x = rng.gen_range(0.0..settings::SCREEN_WIDTH as f32);
            let y = rng.gen_range(0.0..settings::SCREEN_HEIGHT as f32);
            Rc::new(RefCell::new(Prey::new(x, y, &mut rng)))
        })
        .collect();

    // Python: SpatialHash(settings.SCREEN_WIDTH // settings.SIGHT_RANGE_*)
    let cell_pred = ((settings::SCREEN_WIDTH as f32) / settings::SIGHT_RANGE_PREDATOR)
        .floor()
        .max(1.0) as i32;
    let cell_prey = ((settings::SCREEN_WIDTH as f32) / settings::SIGHT_RANGE_PREY)
        .floor()
        .max(1.0) as i32;

    let mut spatial_preds: SpatialHash<Predator> = SpatialHash::new(cell_pred);
    let mut spatial_preys: SpatialHash<Prey> = SpatialHash::new(cell_prey);

    let mut step: i64 = 0;
    let start = Instant::now();

    // ---- Simulation loop ----
    // Python headless_runner iterates over copies to avoid removal-while-iterating bugs.
    // We'll do the same.
    while step < max_steps {
        // extinction
        if predators.is_empty() || preys.is_empty() {
            break;
        }

        // rebuild spatial hashes
        spatial_preds.clear();
        spatial_preys.clear();

        for p in &predators {
            spatial_preds.insert(Rc::clone(p));
        }
        for pr in &preys {
            spatial_preys.insert(Rc::clone(pr));
        }

        // Predators: iterate over snapshot
        let active_predators: Vec<PredatorRc> = predators.iter().cloned().collect();
        for pred_rc in active_predators {
            // if already removed, skip (identity check)
            if !predators.iter().any(|p| Rc::ptr_eq(p, &pred_rc)) {
                continue;
            }

            let (px, py) = {
                let p = pred_rc.borrow();
                (p.x, p.y)
            };

            let nearby_preys = spatial_preys.query(px, py);

            {
                let mut pred = pred_rc.borrow_mut();
                let inputs = pred.get_inputs(&nearby_preys);
                pred.move_step(&inputs);
            }

            let dead = pred_rc.borrow().energy < 0.0;
            if dead {
                if let Some(pos) = predators.iter().position(|p| Rc::ptr_eq(p, &pred_rc)) {
                    predators.remove(pos);
                }
                continue;
            }

            // Python: reproduction commented out in main.py; headless_runner keeps it as pass.
        }

        // Preys: iterate over snapshot
        let active_preys: Vec<PreyRc> = preys.iter().cloned().collect();
        for prey_rc in active_preys {
            if !preys.iter().any(|p| Rc::ptr_eq(p, &prey_rc)) {
                continue;
            }

            let (x, y) = {
                let pr = prey_rc.borrow();
                (pr.x, pr.y)
            };

            let nearby_preds = spatial_preds.query(x, y);

            // IMPORTANT: prey.get_inputs can mutate predators (eating + predator.reproduce)
            let inputs_opt = {
                let pr = prey_rc.borrow();
                pr.get_inputs(&nearby_preds, &mut predators, &mut rng)
            };

            if let Some(inputs) = inputs_opt {
                prey_rc.borrow_mut().move_step(&inputs);
            } else {
                // eaten
                if let Some(pos) = preys.iter().position(|p| Rc::ptr_eq(p, &prey_rc)) {
                    preys.remove(pos);
                }
                continue;
            }

            // Reproduction
            if preys.len() < settings::MAX_PREY_COUNT {
                if let Some(new_prey) = prey_rc.borrow_mut().reproduce(&mut rng) {
                    preys.push(new_prey);
                }
            }
        }

        step += 1;
    }

    let duration = start.elapsed().as_secs_f64();
    let avg_step_time = if step > 0 {
        duration / (step as f64)
    } else {
        0.0
    };

    let survived = !predators.is_empty() && !preys.is_empty();

    SimResult {
        survived,
        steps: step,
        avg_step_time,
        final_predators: predators.len(),
        final_preys: preys.len(),
        parameters: params_raw.clone(),
        error: None,
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
