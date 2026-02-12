// ============================================================================
// BENCHMARK ARENA & COMPLEXITY LOGGER
// ============================================================================
// Loads top predator brains from `top_predators/`, measures their structural
// complexity (neuron & connection counts), and tests each brain in a
// controlled arena to measure hunting effectiveness.
//
// Usage:
//   cargo run --release --bin arena_test              # CSV-only (headless)
//   cargo run --release --bin arena_test -- --visual  # watch each arena live
//
// Output (headless): CSV to stdout
//   Generation,Avg_Kills,Num_Neurons,Num_Connections
// ============================================================================

use predator_vs_prey::brain_neural_network::NeuralNetwork;
use predator_vs_prey::data_manager::Frame;
use predator_vs_prey::game::Game;
use predator_vs_prey::settings;
use predator_vs_prey::visualization::{draw_frame, draw_game_stats};

use macroquad::prelude::*;
use rayon::prelude::*;
use std::fs;
use std::path::PathBuf;

/// Number of prey placed in the arena for each test.
const ARENA_PREY_COUNT: usize = 350;
/// Ticks to simulate per arena run.
const ARENA_TICKS: usize = 2000;
/// How many times to repeat each arena test (results are averaged).
const ARENA_RUNS: usize = 5;
/// Fixed seeds for arena runs – identical across all generations so every
/// top predator faces the exact same prey distributions and behaviour.
const ARENA_SEEDS: [u64; ARENA_RUNS] = [42, 123, 256, 789, 1024];

/// Represents a top predator brain loaded from disk together with its generation.
struct TopPredatorEntry {
    generation: usize,
    rank: usize, // 1-based rank within the generation (1 = best)
    brain: NeuralNetwork,
}

/// Scan `top_predators/` for top predator JSON files, parse generation and rank from
/// filename, and return entries sorted by (generation, rank).
///
/// Supports both old format `gen_5000.json` (treated as rank 1) and new
/// format `gen_5000_rank2.json`.
fn load_top_predators(dir: &str) -> Vec<TopPredatorEntry> {
    let path = PathBuf::from(dir);
    if !path.is_dir() {
        eprintln!(
            "Directory '{}' not found. Run a simulation with --top-predators first.",
            dir
        );
        return Vec::new();
    }

    let mut entries: Vec<TopPredatorEntry> = Vec::new();

    for entry in fs::read_dir(&path).expect("Failed to read top_predators directory") {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();

        if !name.starts_with("gen_") || !name.ends_with(".json") {
            continue;
        }

        // Strip "gen_" prefix and ".json" suffix
        let stem = &name[4..name.len() - 5];

        // Parse generation and optional rank
        let (generation, rank) = if let Some(idx) = stem.find("_rank") {
            let gen: usize = match stem[..idx].parse() {
                Ok(g) => g,
                Err(_) => { eprintln!("Skipping {}", name); continue; }
            };
            let r: usize = match stem[idx + 5..].parse() {
                Ok(r) => r,
                Err(_) => { eprintln!("Skipping {}", name); continue; }
            };
            (gen, r)
        } else {
            // Old format: gen_5000.json → rank 1
            match stem.parse::<usize>() {
                Ok(g) => (g, 1),
                Err(_) => { eprintln!("Skipping {}", name); continue; }
            }
        };

        let contents = match fs::read_to_string(entry.path()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to read {}: {}", name, e);
                continue;
            }
        };

        let brain: NeuralNetwork = match serde_json::from_str(&contents) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Failed to deserialize {}: {}", name, e);
                continue;
            }
        };

        entries.push(TopPredatorEntry { generation, rank, brain });
    }

    entries.sort_by(|a, b| a.generation.cmp(&b.generation).then(a.rank.cmp(&b.rank)));
    entries
}

// ---------------------------------------------------------------------------
//  Arena helpers
// ---------------------------------------------------------------------------

/// Create a fresh arena game with the given top predator brain injected at centre.
fn make_arena_game(brain: &NeuralNetwork, seed: u64) -> Game {
    let mut game = Game::new(
        None,
        0,                // 0 initial predators
        ARENA_PREY_COUNT, // 150 prey
        1,                // max 1 predator (the top predator)
        0,                // max_preys = 0 → no prey reproduction
        seed,
    );
    let cx = settings::SCREEN_WIDTH as f32 / 2.0;
    let cy = settings::SCREEN_HEIGHT as f32 / 2.0;
    game.inject_predator_with_brain(cx, cy, settings::PRED_ENERGY, brain.clone());
    game
}

/// Run a single arena test headlessly and return kill count.
fn run_arena(brain: &NeuralNetwork, seed: u64) -> usize {
    let mut game = make_arena_game(brain, seed);

    for _ in 0..ARENA_TICKS {
        game.next_frame();
    }

    // Use the predator's own kill counter – not prey delta – because
    // prey reproduce and final_prey can exceed the initial count.
    game.total_predator_kills()
}

// ---------------------------------------------------------------------------
//  Headless mode (CSV output)
// ---------------------------------------------------------------------------

fn run_headless(top_predators: &[TopPredatorEntry]) {
    println!("Generation,Rank,Avg_Kills,Num_Neurons,Num_Connections");

    // Collect results in parallel, then print in order
    let results: Vec<_> = top_predators
        .par_iter()
        .map(|entry| {
            let num_neurons = entry.brain.num_neurons();
            let num_connections = entry.brain.num_connections();

            // Run all seeds in parallel for this top predator
            let total_kills: usize = ARENA_SEEDS
                .par_iter()
                .map(|&seed| run_arena(&entry.brain, seed))
                .sum();

            let avg_kills = total_kills as f64 / ARENA_RUNS as f64;

            (entry.generation, entry.rank, avg_kills, num_neurons, num_connections)
        })
        .collect();

    // Print in sorted order (already sorted by load_champions)
    for (gen, rank, avg_kills, neurons, conns) in results {
        println!("{},{},{:.1},{},{}", gen, rank, avg_kills, neurons, conns);
    }
}

// ---------------------------------------------------------------------------
//  Visual mode (macroquad window)
// ---------------------------------------------------------------------------

fn window_conf() -> Conf {
    Conf {
        window_title: "Arena Test – Top Predator Viewer".to_string(),
        window_width: settings::SCREEN_WIDTH + 200, // room for stats panel
        window_height: settings::SCREEN_HEIGHT,
        ..Default::default()
    }
}

/// Visual mode: step through each top predator, watch the arena live,
/// press SPACE to pause/unpause, N/RIGHT to skip to the next top predator,
/// ESC/Q to quit.
async fn run_visual(top_predators: Vec<TopPredatorEntry>) {
    let mut champ_idx: usize = 0;
    let mut game: Option<Game> = None;
    let mut tick: usize = 0;
    let mut paused = false;
    let mut run_index: usize = 0; // which of the ARENA_RUNS we're on
    let mut last_frame: Option<Frame> = None;

    // Per-champion accumulated results
    let mut run_kills: Vec<usize> = Vec::new();

    loop {
        // --- All champions done ---
        if champ_idx >= top_predators.len() {
            break;
        }

        // --- Initialise game for current champion + run ---
        if game.is_none() {
            let seed = ARENA_SEEDS[run_index];
            game = Some(make_arena_game(&top_predators[champ_idx].brain, seed));
            tick = 0;
            // Produce the initial frame so something is visible immediately
            if let Some(ref mut g) = game {
                last_frame = Some(g.next_frame());
                tick = 1;
            }
        }

        // --- Input ---
        if is_key_pressed(KeyCode::Escape) || is_key_pressed(KeyCode::Q) {
            break;
        }
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        // Skip to next top predator (or next run)
        if is_key_pressed(KeyCode::N) || is_key_pressed(KeyCode::Right) {
            run_kills.clear();
            run_index = 0;
            champ_idx += 1;
            game = None;
            last_frame = None;
            next_frame().await;
            continue;
        }

        // --- Simulation step ---
        if !paused && tick < ARENA_TICKS {
            if let Some(ref mut g) = game {
                let frame = g.next_frame();
                last_frame = Some(frame);
                tick += 1;
            }
        }

        // --- Draw (always, even when a run just finished) ---
        clear_background(settings::BACKGROUND_COLOR);

        if let Some(ref frame) = last_frame {
            draw_frame(frame, false, None);

            let (pred_count, prey_count) = frame.counts();
            draw_game_stats(pred_count, prey_count, tick);
        }

        // HUD overlay
        let entry = &top_predators[champ_idx];
        let kills_so_far = if let Some(ref g) = game {
            g.total_predator_kills()
        } else {
            0
        };
        let hud = format!(
            "Gen {} Rank {} | Run {}/{} | Tick {}/{} | Kills {} | Neurons {} | Conns {}{}",
            entry.generation,
            entry.rank,
            run_index + 1,
            ARENA_RUNS,
            tick,
            ARENA_TICKS,
            kills_so_far,
            entry.brain.num_neurons(),
            entry.brain.num_connections(),
            if paused { " [PAUSED]" } else { "" },
        );
        draw_text(&hud, 10.0, 20.0, 22.0, WHITE);
        draw_text(
            "SPACE: pause | N/Right: skip | Q/Esc: quit",
            10.0,
            settings::SCREEN_HEIGHT as f32 - 10.0,
            18.0,
            Color::from_rgba(180, 180, 180, 200),
        );

        next_frame().await;

        // --- If this run finished, record kills and advance AFTER drawing ---
        if tick >= ARENA_TICKS {
            if let Some(ref g) = game {
                run_kills.push(g.total_predator_kills());
            }
            run_index += 1;

            if run_index >= ARENA_RUNS {
                let avg_kills =
                    run_kills.iter().sum::<usize>() as f64 / run_kills.len().max(1) as f64;
                eprintln!(
                    "Gen {:>6} Rank {} | Avg Kills: {:>5.1} | Neurons: {:>3} | Connections: {:>3}",
                    entry.generation,
                    entry.rank,
                    avg_kills,
                    entry.brain.num_neurons(),
                    entry.brain.num_connections(),
                );
                run_kills.clear();
                run_index = 0;
                champ_idx += 1;
            }
            game = None;
            last_frame = None;
        }
    }
}

// ---------------------------------------------------------------------------
//  Entry point
// ---------------------------------------------------------------------------

#[macroquad::main(window_conf)]
async fn main() {
    let visual = std::env::args().any(|a| a == "--visual");

    let top_predators = load_top_predators("top_predators");

    if top_predators.is_empty() {
        eprintln!("No top predator brains found. Nothing to test.");
        return;
    }

    if visual {
        run_visual(top_predators).await;
    } else {
        run_headless(&top_predators);
    }
}
