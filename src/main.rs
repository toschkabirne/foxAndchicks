// src/main.rs

use predator_vs_prey::game::Game;
use predator_vs_prey::settings::{self, DEFAULT_DATA_FILE};
use predator_vs_prey::visualization::{draw_frame, draw_game_stats, window_conf};

use macroquad::prelude::*;

// use pred_prey_sim::{animals::*, settings::*, spatial_hash::*, brain_neural_network::*}

#[macroquad::main(window_conf)]
async fn main() {
    // Get commandline args for filename, else default
    // Minimal CLI parsing (robust, no panics)
    let mut run_live = false;
    let mut draw_sight_lines = true;
    let mut filename: String = DEFAULT_DATA_FILE.to_string();
    let mut total_frames: i32 = settings::DEFAULT_TOTAL_FRAMES;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--run-live" => run_live = true,
            "--no-sight" | "--no-sight-lines" => draw_sight_lines = false,
            "--file" | "-f" => {
                filename = it.next().unwrap_or_else(|| DEFAULT_DATA_FILE.to_string());
            }
            "--frames" | "-n" => {
                total_frames = it.next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(settings::DEFAULT_TOTAL_FRAMES);
            }
            "--help" | "-h" => {
                println!("Usage:");
                println!("  cargo run -- [--run-live] [--no-sight] [--file <path>] [--frames <num>]");
                println!();
                println!("Options:");
                println!("  --run-live           Run simulation with live rendering");
                println!("  --no-sight           Hide sight lines during playback");
                println!("  --file, -f <path>    Specify the output file path");
                println!("  --frames, -n <num>   Number of frames to record");
                println!();
                println!("To playback an existing recording, use:");
                println!("  cargo run --bin playback -- --file <path>");
                println!();
                println!("Examples:");
                println!("  cargo run -- --run-live");
                println!("  cargo run -- --run-live --no-sight");
                println!("  cargo run -- --frames 5000");
                return;
            }
            other => {
                // Convenience: treat unknown single arg as filename
                filename = other.to_string();
            }
        }
    }

    if run_live {
        println!("Running live simulation...");
        playback_live(&filename, draw_sight_lines).await;
    } else {
        record_then_playback(&filename, draw_sight_lines, total_frames).await;
    }
}

async fn record_then_playback(filename: &str, draw_sight_lines: bool, total_frames: i32) {
    let mut game = Game::new_default(filename);

    // Record headless (no rendering)
    for i in 0..total_frames {
        game.calculate_and_store_next_frame();
        if i % 500 == 0 {
            println!("Recording frame {}/{}", i, total_frames);
        }
    }

    // Get the actual filename before dropping (includes timestamp)
    let processed_filename = game.get_data_filename().to_string();

    // IMPORTANT: close/flush the writer before reading the file for playback
    drop(game);

    // Playback with visualization
    Game::playback(&processed_filename, draw_sight_lines).await;
}

async fn playback_live(filename: &str, draw_sight_lines: bool) {
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
