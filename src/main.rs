// src/main.rs
//
// Main binary for running a live simulation with rendering.

use predator_vs_prey::game::Game;
use predator_vs_prey::settings::{self, DEFAULT_DATA_FILE};
use predator_vs_prey::visualization::{draw_frame, draw_game_stats, window_conf};

use macroquad::prelude::*;

#[macroquad::main(window_conf)]
async fn main() {
    let mut draw_sight_lines = true;
    let mut filename: String = DEFAULT_DATA_FILE.to_string();

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--no-sight" | "--no-sight-lines" => draw_sight_lines = false,
            "--file" | "-f" => {
                filename = it.next().unwrap_or_else(|| DEFAULT_DATA_FILE.to_string());
            }
            "--help" | "-h" => {
                println!("Run a live simulation with rendering.");
                println!();
                println!("Usage:");
                println!("  cargo run -- [--no-sight] [--file <path>]");
                println!();
                println!("Options:");
                println!("  --no-sight           Hide sight lines during simulation");
                println!("  --file, -f <path>    Specify the output file path");
                println!();
                println!("Other binaries:");
                println!("  cargo run --bin record -- [--file <path>] [--frames <num>]");
                println!("  cargo run --bin playback -- --file <path>");
                println!();
                println!("Examples:");
                println!("  cargo run");
                println!("  cargo run -- --no-sight");
                println!("  cargo run -- --file simulations/my_sim.bin");
                return;
            }
            other => {
                // Convenience: treat unknown single arg as filename
                filename = other.to_string();
            }
        }
    }

    println!("Running live simulation...");
    run_live(&filename, draw_sight_lines).await;
}

async fn run_live(filename: &str, draw_sight_lines: bool) {
    let mut game = Game::new_default(filename);

    // Run simulation with live rendering
    loop {
        clear_background(settings::BACKGROUND_COLOR);
        let frame = game.next_frame();
        draw_frame(&frame, draw_sight_lines);

        let (pred_count, prey_count) = frame.counts();
        draw_game_stats(pred_count, prey_count, game.frame_count);

        // Optional: break after certain frames or on key press
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        next_frame().await;
    }
}
