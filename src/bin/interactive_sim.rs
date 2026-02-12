use crate::settings::*;
use macroquad::prelude::*;
use std::env;
use std::io::{self, Write};

use predator_vs_prey::animals::{wrapped_distance_abs, AnimalCore, Predator, Prey};
use predator_vs_prey::brain_neural_network::NeuralNetwork;
use predator_vs_prey::visualization::draw_neural_network;
use predator_vs_prey::*;

use ::rand::rngs::StdRng;
use ::rand::{Rng, SeedableRng};

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
    /// Default: prey is NN agent, predator is manually controlled
    TestPrey,
    /// Predator is NN agent, prey is manually controlled
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

// Controlled animal movement (keyboard)
// Merged logic for both predator and prey.
// No energy decay as per user preference for manual control.
fn move_with_keyboard(core: &mut AnimalCore, speed: f32) {
    let mut turn = 0.10; // Averaged turn rate (approx)
    let mut current_speed = speed;

    if is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift) {
        turn *= 1.7;
        current_speed *= 1.6;
    }

    if is_key_down(KeyCode::Left) || is_key_down(KeyCode::A) {
        core.angle -= turn;
    }
    if is_key_down(KeyCode::Right) || is_key_down(KeyCode::D) {
        core.angle += turn;
    }

    let mut speed_factor = 0.0;
    if is_key_down(KeyCode::Up) || is_key_down(KeyCode::W) {
        speed_factor = 1.0;
    } else if is_key_down(KeyCode::Down) || is_key_down(KeyCode::S) {
        speed_factor = -0.7; // Using prey's reverse factor
    }

    if speed_factor != 0.0 {
        core.pos.x += speed_factor * current_speed * core.angle.cos();
        core.pos.y += speed_factor * current_speed * core.angle.sin();

        core.pos.x = core.pos.x.rem_euclid(SCREEN_WIDTH as f32);
        core.pos.y = core.pos.y.rem_euclid(SCREEN_HEIGHT as f32);
    }
}

/// Helper to find a respawn position near the center but away from a threat.
fn get_respawn_pos<R: Rng>(avoid_pos: Vec2, rng: &mut R) -> (f32, f32) {
    let min_dist = PRED_RADIUS + PREY_RADIUS + 30.0;
    let world_w = SCREEN_WIDTH as f32;
    let world_h = SCREEN_HEIGHT as f32;
    let center_x = world_w * 0.5;
    let center_y = world_h * 0.5;
    let offset = 100.0; // Small random offset

    for _ in 0..30 {
        let x = rng.gen_range(center_x - offset..center_x + offset);
        let y = rng.gen_range(center_y - offset..center_y + offset);
        let d = wrapped_distance_abs(avoid_pos, vec2(x, y), world_w, world_h);
        if d >= min_dist {
            return (x, y);
        }
    }

    // Fallback: random position in center area
    let x = rng.gen_range(center_x - offset..center_x + offset);
    let y = rng.gen_range(center_y - offset..center_y + offset);
    (x, y)
}

#[macroquad::main(window_conf)]
async fn main() {
    let mode = parse_test_mode();
    println!("Mode: {:?}", mode);

    let mutations = read_mutations_from_stdin(mode);
    let mut rng = StdRng::seed_from_u64(settings::SEED);

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
                move_with_keyboard(&mut predator.core, PRED_SPEED);

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
                    let (x, y) = get_respawn_pos(predator.core.pos, &mut rng);
                    prey = Prey::new(x, y, &mut rng);

                    let input_len = prey.get_inputs(std::iter::once(&predator)).len();
                    prey.core.brain = NeuralNetwork::new(input_len, 2, mutations, bias(), &mut rng);
                }
            }

            TestMode::TestPredator => {
                // Prey is keyboard dummy
                move_with_keyboard(&mut prey.core, PREY_SPEED);

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
                    println!("Predator caught prey! Respawning NN-predator.");
                    let (x, y) = get_respawn_pos(prey.core.pos, &mut rng);
                    predator = Predator::new(x, y, &mut rng);

                    let input_len = predator.get_inputs(std::iter::once(&prey)).len();
                    predator.core.brain =
                        NeuralNetwork::new(input_len, 2, mutations, bias(), &mut rng);
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
