// src/bin/playback.rs
//
// Standalone binary for playing back recorded simulation files.

use predator_vs_prey::game::Game;
use predator_vs_prey::visualization::window_conf;

#[macroquad::main(window_conf)]
async fn main() {
    let mut draw_sight_lines = true;
    let mut filename: Option<String> = None;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--no-sight" | "--no-sight-lines" => draw_sight_lines = false,
            "--file" | "-f" => {
                filename = it.next();
            }
            "--help" | "-h" => {
                println!("Playback a recorded simulation file.");
                println!();
                println!("Usage:");
                println!("  cargo run --bin playback -- --file <path> [--no-sight]");
                println!();
                println!("Options:");
                println!("  --file, -f <path>    Path to the recording file (required)");
                println!("  --no-sight           Hide sight lines during playback");
                println!();
                println!("Examples:");
                println!("  cargo run --bin playback -- --file simulations/recording.bin");
                println!("  cargo run --bin playback -- -f simulations/recording.bin --no-sight");
                return;
            }
            other => {
                // Convenience: treat unknown single arg as filename
                filename = Some(other.to_string());
            }
        }
    }

    let filename = match filename {
        Some(f) => f,
        None => {
            eprintln!("Error: No recording file specified.");
            eprintln!("Usage: cargo run --bin playback -- --file <path>");
            eprintln!("Run with --help for more information.");
            return;
        }
    };

    println!("Playing back recording: {}", filename);
    Game::playback(&filename, draw_sight_lines).await;
}
