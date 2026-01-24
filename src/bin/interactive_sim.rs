use crate::settings::*;
use ::rand::Rng;
use macroquad::prelude::*;
use std::env;
use std::io::{self, Write};

use predator_vs_prey::animals::{wrapped_distance_abs, Predator, Prey};
use predator_vs_prey::brain_neural_network::NeuralNetwork;
use predator_vs_prey::visualization::draw_neural_network;
use predator_vs_prey::*;

fn window_conf() -> Conf {
    Conf {
        window_title: "Interactive Predator-Prey".to_string(),
        window_width: SCREEN_WIDTH,
        window_height: SCREEN_HEIGHT,
        ..Default::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TestMode {
    /// Default: prey is NN agent, predator is keyboard dummy (you hunt)
    TestPrey,
    /// Predator is NN agent, prey is keyboard dummy (you flee/tease)
    TestPredator,
}

fn parse_test_mode() -> TestMode {
    // Accept: "test-prey" (default), "test-predator"
    // Also tolerate "--test-prey" / "--test-predator"
    let mut mode = TestMode::TestPrey;

    for arg in env::args().skip(1) {
        let a = arg.trim().to_lowercase();
        match a.as_str() {
            "test-prey" | "--test-prey" => mode = TestMode::TestPrey,
            "test-predator" | "--test-predator" => mode = TestMode::TestPredator,
            _ => {}
        }
    }

    mode
}

fn read_mutations_from_stdin(mode: TestMode) -> usize {
    let who = match mode {
        TestMode::TestPrey => "prey (NN agent)",
        TestMode::TestPredator => "predator (NN agent)",
    };

    print!("How many mutations for the {who}? (Default: 12): ");
    let _ = io::stdout().flush();

    let mut s = String::new();
    if io::stdin().read_line(&mut s).is_err() {
        println!("Invalid input. Using default 12.");
        return 12;
    }

    let trimmed = s.trim();
    if trimmed.is_empty() {
        12
    } else {
        match trimmed.parse::<usize>() {
            Ok(v) => v,
            Err(_) => {
                println!("Invalid input. Using default 12.");
                12
            }
        }
    }
}

// Controlled predator movement (keyboard)
fn move_predator_with_keyboard(pred: &mut Predator) {
    let mut turn = 0.08;
    let mut speed = PRED_SPEED;

    if is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift) {
        turn *= 1.7;
        speed *= 1.6;
    }

    if is_key_down(KeyCode::Left) || is_key_down(KeyCode::A) {
        pred.core.angle -= turn;
    }
    if is_key_down(KeyCode::Right) || is_key_down(KeyCode::D) {
        pred.core.angle += turn;
    }

    // energy decay similar-ish to original
    pred.core.energy -= PRED_DEFAULT_DECAY;

    let mut speed_factor = 0.0;
    if is_key_down(KeyCode::Up) || is_key_down(KeyCode::W) {
        speed_factor = 1.0;
    } else if is_key_down(KeyCode::Down) || is_key_down(KeyCode::S) {
        speed_factor = -0.6;
    }

    if speed_factor != 0.0 {
        pred.core.pos.x += speed_factor * speed * pred.core.angle.cos();
        pred.core.pos.y += speed_factor * speed * pred.core.angle.sin();

        pred.core.pos.x = pred.core.pos.x.rem_euclid(SCREEN_WIDTH as f32);
        pred.core.pos.y = pred.core.pos.y.rem_euclid(SCREEN_HEIGHT as f32);

        pred.core.energy -= (speed_factor * speed_factor) * PRED_MOVING_DECAY;
    }
}

// Controlled prey movement (keyboard dummy)
fn move_prey_with_keyboard(prey: &mut Prey) {
    let mut turn = 0.10;
    let mut speed = PREY_SPEED;

    if is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift) {
        turn *= 1.7;
        speed *= 1.6;
    }

    if is_key_down(KeyCode::Left) || is_key_down(KeyCode::A) {
        prey.core.angle -= turn;
    }
    if is_key_down(KeyCode::Right) || is_key_down(KeyCode::D) {
        prey.core.angle += turn;
    }

    let mut speed_factor = 0.0;
    if is_key_down(KeyCode::Up) || is_key_down(KeyCode::W) {
        speed_factor = 1.0;
    } else if is_key_down(KeyCode::Down) || is_key_down(KeyCode::S) {
        speed_factor = -0.7;
    }

    if speed_factor != 0.0 {
        prey.core.pos.x += speed_factor * speed * prey.core.angle.cos();
        prey.core.pos.y += speed_factor * speed * prey.core.angle.sin();

        prey.core.pos.x = prey.core.pos.x.rem_euclid(SCREEN_WIDTH as f32);
        prey.core.pos.y = prey.core.pos.y.rem_euclid(SCREEN_HEIGHT as f32);
    }
}

/// Respawn prey away from predator so it doesn't instantly collide again.
/// Works for both dummy-prey and NN-prey.
fn respawn_prey_away_from_predator<R: Rng>(prey: &mut Prey, predator: &Predator, rng: &mut R) {
    // Try a few random positions until far enough away
    let min_dist = 250.0_f32.max(PRED_RADIUS + PREY_RADIUS + 30.0);
    let world_w = SCREEN_WIDTH as f32;
    let world_h = SCREEN_HEIGHT as f32;

    for _ in 0..30 {
        let x = rng.gen_range(0.0..world_w);
        let y = rng.gen_range(0.0..world_h);
        let d = wrapped_distance_abs(predator.core.pos, vec2(x, y), world_w, world_h);
        if d >= min_dist {
            *prey = Prey::new(x, y, rng);
            return;
        }
    }

    // Fallback: just place it opposite-ish
    let x = (predator.core.pos.x + world_w * 0.5).rem_euclid(world_w);
    let y = (predator.core.pos.y + world_h * 0.5).rem_euclid(world_h);
    *prey = Prey::new(x, y, rng);
}

#[macroquad::main(window_conf)]
async fn main() {
    let mode = parse_test_mode();
    println!("Mode: {:?}", mode);

    let mutations = read_mutations_from_stdin(mode);
    let mut rng = ::rand::thread_rng();

    let center_x = (SCREEN_WIDTH / 2) as f32;
    let center_y = (SCREEN_HEIGHT / 2) as f32;

    // Spawn entities depending on mode:
    // - NN agent goes to center
    // - keyboard dummy goes offset so it doesn't instantly collide
    let (mut prey, mut predator) = match mode {
        TestMode::TestPrey => {
            // Prey NN in center, predator keyboard offset
            let prey = Prey::new(center_x, center_y, &mut rng);
            let mut predator = Predator::new(center_x + 220.0, center_y, &mut rng);
            predator.core.angle = std::f32::consts::PI; // face center
            (prey, predator)
        }
        TestMode::TestPredator => {
            // Predator NN in center, prey keyboard offset
            let prey = Prey::new(center_x + 220.0, center_y, &mut rng);
            let mut predator = Predator::new(center_x, center_y, &mut rng);
            predator.core.angle = 0.0;
            (prey, predator)
        }
    };

    // Build NN for the active agent using correct input size (derived from get_inputs length)
    match mode {
        TestMode::TestPrey => {
            let input_len = prey.get_inputs(std::iter::once(&predator)).len();
            prey.core.brain = NeuralNetwork::new(input_len, 2, mutations, bias(), &mut rng);
        }
        TestMode::TestPredator => {
            let input_len = predator.get_inputs(std::iter::once(&prey)).len();
            predator.core.brain = NeuralNetwork::new(input_len, 2, mutations, bias(), &mut rng);
        }
    }

    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        // --- Update movement + logic depending on mode ---
        match mode {
            TestMode::TestPrey => {
                // Predator is keyboard dummy
                move_predator_with_keyboard(&mut predator);

                // Prey is NN agent
                let inputs = prey.get_inputs(std::iter::once(&predator));
                prey.move_step(&inputs);

                // If predator catches prey -> respawn prey + rebuild prey brain
                let eat_r = PRED_RADIUS + PREY_RADIUS;
                let d = wrapped_distance_abs(
                    predator.core.pos,
                    prey.core.pos,
                    SCREEN_WIDTH as f32,
                    SCREEN_HEIGHT as f32,
                );

                if d < eat_r {
                    println!("Prey eaten! Respawning NN-prey.");
                    respawn_prey_away_from_predator(&mut prey, &predator, &mut rng);

                    let input_len = prey.get_inputs(std::iter::once(&predator)).len();
                    prey.core.brain = NeuralNetwork::new(input_len, 2, mutations, bias(), &mut rng);
                }
            }

            TestMode::TestPredator => {
                // Prey is keyboard dummy
                move_prey_with_keyboard(&mut prey);

                // Predator is NN agent
                let inputs = predator.get_inputs(std::iter::once(&prey));
                predator.move_step(&inputs);

                // If predator catches prey -> respawn dummy prey (no new brain)
                let eat_r = PRED_RADIUS + PREY_RADIUS;
                let d = wrapped_distance_abs(
                    predator.core.pos,
                    prey.core.pos,
                    SCREEN_WIDTH as f32,
                    SCREEN_HEIGHT as f32,
                );

                if d < eat_r {
                    println!("Prey eaten! Respawning dummy prey.");
                    respawn_prey_away_from_predator(&mut prey, &predator, &mut rng);
                }
            }
        }

        // --- Drawing ---
        clear_background(BLACK);
        visualization::draw_predator(predator.core.pos, predator.core.angle, true);
        visualization::draw_prey(prey.core.pos, prey.core.angle, true);

        // Draw neural network overlay for whichever is the active NN agent
        let nn_ref = match mode {
            TestMode::TestPrey => &prey.core.brain,
            TestMode::TestPredator => &predator.core.brain,
        };
        draw_neural_network(nn_ref, 10.0, 10.0, 240.0, 320.0);

        next_frame().await;
    }
}
