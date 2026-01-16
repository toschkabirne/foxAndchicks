// src/main.rs

use predator_vs_prey::game::Game;
use predator_vs_prey::settings::{self, DEFAULT_DATA_FILE};

use macroquad::prelude::*;
use std::time::Instant;

// use pred_prey_sim::{animals::*, settings::*, spatial_hash::*, brain_neural_network::*};

fn window_conf() -> Conf {
    Conf {
        window_title: "Predator and Prey Simulation".to_string(),
        window_width: settings::SCREEN_WIDTH,
        window_height: settings::SCREEN_HEIGHT,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    // macroquad sets target fps implicitly via frame time, but let's see if we can just cap it or rely on vsync.
    // Rust macroquad doesn't have set_target_fps exposed directly in prelude in recent versions?
    // Checking docs: 'set_target_fps' is not in standard macroquad 0.4 prelude?
    // Actually, usually users just use window conf or await next_frame().
    // But the error said `set_target_fps` not found.
    // We can try to remove it or find where it is.
    // For now I will comment it out as it is not critical for basic wiring, or use a proper alternative if I knew one.
    // However, I will try to leave it out for now to satisfy compilation.
    // set_target_fps(settings::FRAMES_PER_SECOND as u32);

    // Get commandline args for filename, else default
    let args: Vec<String> = std::env::args().collect();
    let run_live = args[1] == "--run-live";
    let filename = if args.len() > 2 {
        &args[2]
    } else {
        DEFAULT_DATA_FILE
    };

    if run_live {
        println!("Running live simulation...");
        playback_live().await;
    } else {
        record_then_playback(filename).await;
    }
}

async fn record_then_playback(filename: &str) {
    let mut game = Game::new_default(filename);

    let start_time = Instant::now();

    // Record headless (no rendering)
    let total_frames = settings::FRAMES_PER_SECOND * 10;
    for i in 0..total_frames {
        game.calculate_and_store_next_frame();
        if i % 500 == 0 {
            println!("Recording frame {}/{}", i, total_frames);
        }
    }

    let elapsed = start_time.elapsed();
    println!("Recording completed in {:.2?}.", elapsed);

    // Playback with visualization
    Game::playback("simulation_data.bin").await;
}

async fn playback_live() {
    let mut game = Game::new_default("simulation_data.bin");

    // Run simulation with live rendering
    loop {
        clear_background(settings::BACKGROUND_COLOR);
        let frame = game.next_frame();
        frame.draw(true);

        // Optional: break after certain frames or on key press
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        next_frame().await;
    }
}
