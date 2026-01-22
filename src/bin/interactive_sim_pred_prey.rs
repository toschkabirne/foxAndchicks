use ::rand::thread_rng;
// use ::rand::Rng;
use crate::settings::*;
use macroquad::prelude::*;
use std::io::{self, Write};

use predator_vs_prey::animals::{Predator, Prey, wrapped_distance_abs};
use predator_vs_prey::brain_neural_network::NeuralNetwork;
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

fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color::from_rgba(r, g, b, 255)
}

// draw_neural_network(screen, prey.brain, 10, 10, 240, 320) in macroquad
fn draw_neural_network(nn: &NeuralNetwork, x: f32, y: f32, width: f32, height: f32) {
    if nn.last_inputs.is_empty() {
        return;
    }
    let inputs = &nn.last_inputs;

    if nn.last_activations.is_empty() {
        return;
    }
    let activations = &nn.last_activations;

    // Frame & background (Python: white fill + black frame)
    draw_rectangle(x, y, width, height, WHITE);
    draw_rectangle_lines(x, y, width, height, 2.0, BLACK);

    let num_inputs = nn.num_inputs;
    let num_outputs = nn.num_outputs;
    let total_neurons = nn.neuron_number;
    let num_hidden = total_neurons - (num_inputs + 1 + num_outputs);

    // positions
    let mut node_pos: Vec<(f32, f32)> = vec![(0.0, 0.0); total_neurons];

    // Inputs (left)
    for i in 0..num_inputs {
        let px = x + 20.0;
        let py = y + (height / (num_inputs as f32 + 1.0)) * (i as f32 + 1.0);
        node_pos[i] = (px, py);
    }

    // Bias (left, below inputs)
    let bias_id = num_inputs;
    node_pos[bias_id] = (
        x + 20.0,
        y + (height / (num_inputs as f32 + 1.0)) * (num_inputs as f32 + 1.0),
    );

    let in_bi = num_inputs + 1;

    // Outputs (right)
    for i in 0..num_outputs {
        let id = in_bi + i;
        let px = x + width - 20.0;
        let py = y + (height / (num_outputs as f32 + 1.0)) * (i as f32 + 1.0);
        node_pos[id] = (px, py);
    }

    // Hidden (middle)
    for i in 0..num_hidden {
        let id = in_bi + num_outputs + i;
        let offset_x = (width / 2.0) + (((i % 3) as i32 - 1) as f32) * 30.0;
        let px = x + offset_x;
        let py = y + (height / (num_hidden as f32 + 1.0)) * (i as f32 + 1.0);
        node_pos[id] = (px, py);
    }

    // helper activation like python
    let get_activation = |id: usize| -> f32 {
        if id < num_inputs {
            inputs[id]
        } else if id == num_inputs {
            1.0 // python: bias node drawn as 1.0
        } else {
            activations[id - in_bi]
        }
    };

    // Draw edges from Input_Matrix: target rows correspond to (outputs+hidden)
    // Python: rows, cols = nn.Input_Matrix.shape
    for r in 0..nn.input_matrix.len() {
        for c in 0..nn.input_matrix[r].len() {
            let weight = nn.input_matrix[r][c];
            if weight == 0.0 {
                continue;
            }

            let source_id = c;
            let target_id = in_bi + r;

            let source_val = get_activation(source_id);
            let val = source_val * weight;

            let (sx, sy) = node_pos[source_id];
            let (tx, ty) = node_pos[target_id];

            if source_val > 0.0 {
                let intensity = (val.abs() * 255.0).min(255.0) as u8;
                let color = if val > 0.0 {
                    rgb(0, intensity, 0)
                } else {
                    rgb(intensity, 0, 0)
                };
                let w = if intensity > 50 { 2.0 } else { 1.0 };
                draw_line(sx, sy, tx, ty, w, color);
            } else {
                draw_line(sx, sy, tx, ty, 1.0, rgb(200, 200, 200));
            }
        }
    }

    // Draw edges from Hidden_Matrix
    for r in 0..nn.hidden_matrix.len() {
        for c in 0..nn.hidden_matrix[r].len() {
            let weight = nn.hidden_matrix[r][c];
            if weight == 0.0 {
                continue;
            }

            let source_id = in_bi + c;
            let target_id = in_bi + r;

            let source_val = get_activation(source_id);
            let val = source_val * weight;

            let (sx, sy) = node_pos[source_id];
            let (tx, ty) = node_pos[target_id];

            if source_val > 0.0 {
                let intensity = (val.abs() * 255.0).min(255.0) as u8;
                let color = if val > 0.0 {
                    rgb(0, intensity, 0)
                } else {
                    rgb(intensity, 0, 0)
                };
                let w = if intensity > 50 { 2.0 } else { 1.0 };
                draw_line(sx, sy, tx, ty, w, color);
            } else {
                draw_line(sx, sy, tx, ty, 1.0, rgb(200, 200, 200));
            }
        }
    }

    // Draw nodes
    for id in 0..total_neurons {
        let val = get_activation(id);
        let clamped = val.max(-1.0).min(1.0);
        let c_val = (clamped * 255.0) as i32;

        let mut color = if c_val > 0 {
            rgb(0, c_val as u8, 0)
        } else {
            rgb(c_val.abs() as u8, 0, 0)
        };

        if c_val.abs() < 20 {
            color = rgb(150, 150, 150);
        }

        let (px, py) = node_pos[id];
        draw_circle(px, py, 5.0, color);
        draw_circle_lines(px, py, 5.0, 1.0, BLACK);
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
    prey.core.brain = NeuralNetwork::new(NUMBER_SIGHTS_PREY, 2, mutations, bias(), &mut rng);

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
                NeuralNetwork::new(NUMBER_SIGHTS_PREY, 2, mutations, bias(), &mut rng);
        } else {
            // Inputs: prey senses predator without any allocations / Rc / RefCell
            let inputs = prey.sense_predators(std::iter::once(&predator));
            prey.move_step(&inputs);
        }

        // Drawing
        clear_background(BLACK);
        predator.draw();
        prey.draw();

        // Draw neural network overlay
        let nn_ref = &prey.core.brain;
        draw_neural_network(nn_ref, 10.0, 10.0, 240.0, 320.0);

        next_frame().await;
    }
}
