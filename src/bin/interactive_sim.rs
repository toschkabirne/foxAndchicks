use ::rand::thread_rng;
// use ::rand::Rng;
use crate::settings::*;
use macroquad::prelude::*;
use std::io::{self, Write};

use predator_vs_prey::animals::{Predator, Prey, wrapped_distance_abs};
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

fn read_mutations_from_stdin() -> usize {
    // Prompt for mutation count (wie Python)
    print!("How many mutations for the prey? (Default: 12): ");
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

// Controlled predator movement (ersetzt ControlledPredator.move(keys))
fn move_predator_with_keyboard(pred: &mut Predator) {
    // Simple controls:
    // Left/Right: rotate
    // Up: move forward
    // Down: move backward a bit
    // Shift: faster
    let mut turn = 0.08;
    let mut speed = PREDATOR_SPEED;

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

    // Energy decay ähnlich Predator.move_step, aber speed_factor “manuell”
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

        // wrap_position analog zu animals.rs
        pred.core.pos.x = pred.core.pos.x.rem_euclid(SCREEN_WIDTH as f32);
        pred.core.pos.y = pred.core.pos.y.rem_euclid(SCREEN_HEIGHT as f32);

        // Quadratic-ish movement decay (wie im Original, nur mit manuellem speed_factor)
        pred.core.energy -= (speed_factor * speed_factor) * PRED_MOVING_DECAY;
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mutations = read_mutations_from_stdin();

    let mut rng = thread_rng();

    // Spawn entities
    let mut prey = Prey::new(
        (SCREEN_WIDTH / 2) as f32,
        (SCREEN_HEIGHT / 2) as f32,
        &mut rng,
    );

    let mut predator = Predator::new(
        (SCREEN_WIDTH / 2 + 200) as f32,
        (SCREEN_HEIGHT / 2) as f32,
        &mut rng,
    );

    // Face left
    predator.core.angle = std::f32::consts::PI;

    // Apply user-chosen mutations to prey brain
    prey.core.brain = NeuralNetwork::new(PREY_SIGHT_COUNT, 2, mutations, bias(), &mut rng);

    // set_target_fps(FRAMES_PER_SECOND as u32);

    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        // Controlled predator movement
        move_predator_with_keyboard(&mut predator);

        // "Eaten" check: simple collision (Predator circle intersects Prey circle)
        let eat_r = PREDATOR_RADIUS + PREY_RADIUS;
        if wrapped_distance_abs(predator.core.pos, prey.core.pos, SCREEN_WIDTH as f32, SCREEN_HEIGHT as f32) < eat_r {
            println!("Prey eaten! Spawning new prey.");
            prey = Prey::new(
                (SCREEN_WIDTH / 2) as f32,
                (SCREEN_HEIGHT / 2) as f32,
                &mut rng,
            );
            prey.core.brain =
                NeuralNetwork::new(PREY_SIGHT_COUNT, 2, mutations, bias(), &mut rng);
        } else {
            // Inputs: prey senses predator without any allocations / Rc / RefCell
            let inputs = prey.sense_predators(std::iter::once(&predator));
            prey.move_step(&inputs);
        }

        // Drawing
        clear_background(BLACK);
        visualization::draw_predator(predator.core.pos, predator.core.angle, true);
        visualization::draw_prey(prey.core.pos, prey.core.angle, true);

        // Draw neural network overlay
        let nn_ref = &prey.core.brain;
        draw_neural_network(nn_ref, 10.0, 10.0, 240.0, 320.0);

        next_frame().await;
    }
}
