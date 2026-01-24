// src/main.rs
//
// Main binary for running a live simulation with rendering.

use predator_vs_prey::data_manager::AnimalType;
use predator_vs_prey::game::Game;
use predator_vs_prey::settings::{self, DEFAULT_DATA_FILE};
use predator_vs_prey::visualization::{
    draw_frame, draw_game_stats, draw_neural_network, window_conf,
};

use macroquad::prelude::*;

#[macroquad::main(window_conf)]
async fn main() {
    let mut draw_sight_lines = true;
    let mut filename: Option<String> = None;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--no-sight" | "--no-sight-lines" => draw_sight_lines = false,
            "--file" | "-f" => {
                filename = Some(it.next().unwrap_or_else(|| DEFAULT_DATA_FILE.to_string()));
            }
            "--help" | "-h" => {
                println!("Run a live simulation with rendering.");
                println!();
                println!("Usage:");
                println!("  cargo run -- [--no-sight] [--file <path>]");
                println!();
                println!("Options:");
                println!("  --no-sight           Hide sight lines during simulation");
                println!("  --file, -f <path>    Specify the output file path (if omitted, no data is saved)");
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
                eprintln!("Unknown argument: {}", other);
            }
        }
    }

    println!("Running live simulation...");
    run_live(filename.as_deref(), draw_sight_lines).await;
}

async fn run_live(filename: Option<&str>, draw_sight_lines: bool) {
    let mut game = Game::new_default(filename);
    let mut selected_animal: Option<(AnimalType, usize)> = None;

    // Run simulation with live rendering
    loop {
        clear_background(settings::BACKGROUND_COLOR);

        // Handle mouse input to select/deselect animals
        if is_mouse_button_pressed(MouseButton::Left) {
            let (mx, my) = mouse_position();
            if let Some(target) = game.get_closest_animal_at(mx, my) {
                selected_animal = Some(target);
            } else {
                selected_animal = None;
            }
        }

        let frame = game.next_frame();
        draw_frame(&frame, draw_sight_lines);

        // Draw neural network of selected animal
        if let Some((atype, id)) = selected_animal {
            if let Some(brain) = game.get_animal_brain(atype, id) {
                draw_neural_network(brain, 10.0, 10.0, 240.0, 320.0);
            } else {
                // Selected animal has died (or was removed)
                selected_animal = None;
            }
        }

        let (pred_count, prey_count) = frame.counts();
        draw_game_stats(pred_count, prey_count, game.frame_count);

        // Optional: break after certain frames or on key press
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        next_frame().await;
    }
}
