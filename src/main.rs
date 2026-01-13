// src/main.rs

use predatorVsPrey::animals::{Predator, Prey};
use predatorVsPrey::game::Game;
use predatorVsPrey::settings;

use macroquad::prelude::*;

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


    let mut game = predatorVsPrey::game::Game::new_default(
        "simulation_data.bin",
    );

    for _ in 0..10000 {
        game.next_frame();
    }

    Game::playback("simulation_data.bin").await;
}
