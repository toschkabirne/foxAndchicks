// src/bin/record.rs
//
// Standalone binary for recording simulation data without rendering.

use predator_vs_prey::game::Game;
use predator_vs_prey::settings::{self, DEFAULT_DATA_FILE};

fn main() {
    let mut filename: String = DEFAULT_DATA_FILE.to_string();
    let mut total_frames: i32 = settings::DEFAULT_TOTAL_FRAMES;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--file" | "-f" => {
                filename = it.next().unwrap_or_else(|| DEFAULT_DATA_FILE.to_string());
            }
            "--frames" | "-n" => {
                total_frames = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(settings::DEFAULT_TOTAL_FRAMES);
            }
            "--help" | "-h" => {
                println!("Record a simulation to a file (headless, no rendering).");
                println!();
                println!("Usage:");
                println!("  cargo run --bin record -- [--file <path>] [--frames <num>]");
                println!();
                println!("Options:");
                println!("  --file, -f <path>    Specify the output file path");
                println!("  --frames, -n <num>   Number of frames to record");
                println!();
                println!("Examples:");
                println!("  cargo run --bin record -- --frames 5000");
                println!("  cargo run --bin record -- -f simulations/my_recording.bin -n 10000");
                println!();
                println!("To playback an existing recording, use:");
                println!("  cargo run --bin playback -- --file <path>");
                return;
            }
            other => {
                // Convenience: treat unknown single arg as filename
                filename = other.to_string();
            }
        }
    }

    record(&filename, total_frames);
}

fn record(filename: &str, total_frames: i32) {
    let mut game = Game::new_default(Some(filename));

    // Record headless (no rendering)
    for i in 0..total_frames {
        game.calculate_and_store_next_frame();
        if i % 500 == 0 {
            println!("Recording frame {}/{}", i, total_frames);
        }
    }

    // Extract filename before dropping game (which closes the file)
    let filename_with_timestamp = game
        .get_data_filename()
        .expect("DataManager should be present")
        .to_string();

    // IMPORTANT: close/flush the writer before reading the file for playback
    drop(game);

    println!(
        "Recording complete and saved to: {}",
        filename_with_timestamp
    );
    println!(
        "Playback using: cargo run --bin playback -- --file {}",
        filename_with_timestamp
    );
}
