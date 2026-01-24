// src/visualization.rs
//
// Visualization utilities for rendering the simulation

use crate::brain_neural_network::NeuralNetwork;
use crate::data_manager::{AnimalType, Frame};
use crate::settings::{self};
use macroquad::prelude::*;

pub const STATS_PANEL_WIDTH: i32 = 200;
pub const PLAYBACK_CONTROLS_HEIGHT: f32 = 60.0;
pub const NOSE_LENGTH: f32 = 6.0;

/// Returns the window configuration for macroquad
pub fn window_conf() -> Conf {
    Conf {
        window_title: "Predator and Prey Simulation".to_string(),
        window_width: settings::SCREEN_WIDTH + STATS_PANEL_WIDTH,
        window_height: settings::SCREEN_HEIGHT,
        ..Default::default()
    }
}

/// Draw a line that wraps around the toroidal world.
/// If the line crosses a border, it draws the wrapped portion on the opposite side.
pub fn draw_wrapped_line(
    start: Vec2,
    end: Vec2,
    width: f32,
    height: f32,
    thickness: f32,
    color: Color,
) {
    // Draw the main line (possibly going outside bounds)
    draw_line(start.x, start.y, end.x, end.y, thickness, color);

    // Check if end point is outside bounds and draw wrapped segments
    let mut offsets = Vec::new();

    if end.x < 0.0 {
        offsets.push(vec2(width, 0.0));
    } else if end.x >= width {
        offsets.push(vec2(-width, 0.0));
    }

    if end.y < 0.0 {
        offsets.push(vec2(0.0, height));
    } else if end.y >= height {
        offsets.push(vec2(0.0, -height));
    }

    // Draw offset copies of the line for wrapping
    for offset in &offsets {
        draw_line(
            start.x + offset.x,
            start.y + offset.y,
            end.x + offset.x,
            end.y + offset.y,
            thickness,
            color,
        );
    }

    // Handle corner case (both x and y wrap)
    if offsets.len() == 2 {
        let corner_offset = offsets[0] + offsets[1];
        draw_line(
            start.x + corner_offset.x,
            start.y + corner_offset.y,
            end.x + corner_offset.x,
            end.y + corner_offset.y,
            thickness,
            color,
        );
    }
}

/// Draws the statistics panel on the right side of the game field
pub fn draw_game_stats(pred_count: usize, prey_count: usize, frame_count: usize) {
    let panel_x = settings::SCREEN_WIDTH as f32;
    draw_rectangle(
        panel_x,
        0.0,
        STATS_PANEL_WIDTH as f32,
        settings::SCREEN_HEIGHT as f32,
        Color::from_rgba(30, 30, 30, 255),
    );
    draw_line(
        panel_x,
        0.0,
        panel_x,
        settings::SCREEN_HEIGHT as f32,
        2.0,
        WHITE,
    );

    let text_x = panel_x + 15.0;
    draw_text("Statistics", text_x, 30.0, 24.0, WHITE);

    let text_pred = format!("Predators: {}", pred_count);
    let text_prey = format!("Preys: {}", prey_count);
    let frame_text = format!("Frame: {}", frame_count);

    draw_text(&text_pred, text_x, 70.0, 20.0, settings::PRED_COLOR);
    draw_text(&text_prey, text_x, 95.0, 20.0, settings::PREY_COLOR);
    draw_text(&frame_text, text_x, 130.0, 20.0, WHITE);
}

/// Draws all animals in the given frame
pub fn draw_frame(frame: &Frame, draw_sight_lines: bool) {
    for animal in &frame.animals {
        let pos = vec2(animal.x, animal.y);
        match animal.animal_type {
            AnimalType::Predator => {
                draw_predator(pos, animal.angle, draw_sight_lines);
            }
            AnimalType::Prey => {
                draw_prey(pos, animal.angle, draw_sight_lines);
            }
        }
    }
}

/// State for the playback slider
pub struct PlaybackState {
    pub current_frame: usize,
    pub is_playing: bool,
    pub playback_speed: f32,
    pub is_dragging: bool,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self {
            current_frame: 0,
            is_playing: true,
            playback_speed: 1.0,
            is_dragging: false,
        }
    }
}

/// Draws playback controls including a slider and play/pause button.
/// Returns the new frame index if the user interacted with the slider.
pub fn draw_playback_controls(state: &mut PlaybackState, total_frames: usize) {
    if total_frames == 0 {
        return;
    }

    let panel_y = settings::SCREEN_HEIGHT as f32 - PLAYBACK_CONTROLS_HEIGHT;
    let panel_width = settings::SCREEN_WIDTH as f32;

    // Draw background panel
    draw_rectangle(
        0.0,
        panel_y,
        panel_width,
        PLAYBACK_CONTROLS_HEIGHT,
        Color::from_rgba(40, 40, 40, 230),
    );
    draw_line(0.0, panel_y, panel_width, panel_y, 2.0, WHITE);

    // Slider dimensions
    let slider_margin = 20.0;
    let slider_x = slider_margin + 50.0; // Leave room for play button
    let slider_y = panel_y + 25.0;
    let slider_width = panel_width - 2.0 * slider_margin - 60.0;
    let slider_height = 8.0;
    let handle_radius = 10.0;

    // Draw play/pause button
    let button_x = slider_margin;
    let button_y = panel_y + 15.0;
    let button_size = 30.0;

    let mouse_pos = mouse_position();
    let mouse_in_button = mouse_pos.0 >= button_x
        && mouse_pos.0 <= button_x + button_size
        && mouse_pos.1 >= button_y
        && mouse_pos.1 <= button_y + button_size;

    let button_color = if mouse_in_button {
        Color::from_rgba(100, 100, 100, 255)
    } else {
        Color::from_rgba(70, 70, 70, 255)
    };

    draw_rectangle(button_x, button_y, button_size, button_size, button_color);
    draw_rectangle_lines(button_x, button_y, button_size, button_size, 2.0, WHITE);

    // Draw play or pause icon
    if state.is_playing {
        // Pause icon (two vertical bars)
        let bar_width = 6.0;
        let bar_height = 16.0;
        let bar_y = button_y + (button_size - bar_height) / 2.0;
        draw_rectangle(button_x + 8.0, bar_y, bar_width, bar_height, WHITE);
        draw_rectangle(button_x + 16.0, bar_y, bar_width, bar_height, WHITE);
    } else {
        // Play icon (triangle)
        let cx = button_x + button_size / 2.0 + 2.0;
        let cy = button_y + button_size / 2.0;
        draw_triangle(
            Vec2::new(cx - 8.0, cy - 8.0),
            Vec2::new(cx - 8.0, cy + 8.0),
            Vec2::new(cx + 8.0, cy),
            WHITE,
        );
    }

    // Handle play/pause button click
    if mouse_in_button && is_mouse_button_pressed(MouseButton::Left) {
        state.is_playing = !state.is_playing;
    }

    // Draw slider track
    draw_rectangle(
        slider_x,
        slider_y,
        slider_width,
        slider_height,
        Color::from_rgba(80, 80, 80, 255),
    );

    // Calculate handle position
    let progress = state.current_frame as f32 / (total_frames - 1).max(1) as f32;
    let handle_x = slider_x + progress * slider_width;
    let handle_y = slider_y + slider_height / 2.0;

    // Draw filled portion of slider
    draw_rectangle(
        slider_x,
        slider_y,
        progress * slider_width,
        slider_height,
        Color::from_rgba(100, 180, 255, 255),
    );

    // Check if mouse is over slider area
    let mouse_in_slider = mouse_pos.0 >= slider_x
        && mouse_pos.0 <= slider_x + slider_width
        && mouse_pos.1 >= slider_y - handle_radius
        && mouse_pos.1 <= slider_y + slider_height + handle_radius;

    // Handle slider interaction
    if is_mouse_button_pressed(MouseButton::Left) && mouse_in_slider {
        state.is_dragging = true;
    }

    if is_mouse_button_released(MouseButton::Left) {
        state.is_dragging = false;
    }

    if state.is_dragging {
        let new_progress = ((mouse_pos.0 - slider_x) / slider_width).clamp(0.0, 1.0);
        state.current_frame = (new_progress * (total_frames - 1) as f32).round() as usize;
    }

    // Draw handle
    let handle_color = if state.is_dragging || mouse_in_slider {
        Color::from_rgba(150, 210, 255, 255)
    } else {
        Color::from_rgba(100, 180, 255, 255)
    };
    draw_circle(handle_x, handle_y, handle_radius, handle_color);
    draw_circle_lines(handle_x, handle_y, handle_radius, 2.0, WHITE);

    // Draw frame counter
    let frame_text = format!("{} / {}", state.current_frame + 1, total_frames);
    let text_width = measure_text(&frame_text, None, 16, 1.0).width;
    draw_text(
        &frame_text,
        slider_x + slider_width / 2.0 - text_width / 2.0,
        panel_y + 52.0,
        16.0,
        WHITE,
    );

    // Draw speed indicator in stats panel
    let speed_text = format!("Speed: {:.1}x", state.playback_speed);
    draw_text(
        &speed_text,
        settings::SCREEN_WIDTH as f32 + 15.0,
        170.0,
        18.0,
        WHITE,
    );

    // Handle keyboard controls
    if is_key_pressed(KeyCode::Space) {
        state.is_playing = !state.is_playing;
    }

    if is_key_pressed(KeyCode::Left) && !state.is_dragging {
        if state.current_frame > 0 {
            state.current_frame -= 1;
        }
        state.is_playing = false;
    }

    if is_key_pressed(KeyCode::Right) && !state.is_dragging {
        if state.current_frame < total_frames - 1 {
            state.current_frame += 1;
        }
        state.is_playing = false;
    }

    // Speed controls
    if is_key_pressed(KeyCode::Up) {
        state.playback_speed = (state.playback_speed * 1.5).min(8.0);
    }

    if is_key_pressed(KeyCode::Down) {
        state.playback_speed = (state.playback_speed / 1.5).max(0.1);
    }

    // Jump to start/end
    if is_key_pressed(KeyCode::Home) {
        state.current_frame = 0;
        state.is_playing = false;
    }

    if is_key_pressed(KeyCode::End) {
        state.current_frame = total_frames - 1;
        state.is_playing = false;
    }
}

pub fn draw_predator_sight(pos: Vec2, angle: f32) {
    let n = settings::PRED_SIGHT_COUNT.max(1);
    let world_w = settings::SCREEN_WIDTH as f32;
    let world_h = settings::SCREEN_HEIGHT as f32;

    let fov_rad = settings::PRED_SIGHT_FOV.to_radians();
    let half_fov = fov_rad / 2.0;
    let start_angle = angle - half_fov;
    let end_angle = angle + half_fov;

    for i in 0..n {
        let t = if n > 1 {
            i as f32 / (n as f32 - 1.0)
        } else {
            0.0
        };
        let sight_angle = start_angle + t * (end_angle - start_angle);

        let end_x = pos.x + settings::PRED_SIGHT_RANGE * sight_angle.cos();
        let end_y = pos.y + settings::PRED_SIGHT_RANGE * sight_angle.sin();

        draw_wrapped_line(pos, vec2(end_x, end_y), world_w, world_h, 1.0, YELLOW);
    }
}
pub fn draw_nose(pos: Vec2, angle: f32, radius: f32, color: Color) {
    let end_x = pos.x + (NOSE_LENGTH + radius) * angle.cos();
    let end_y = pos.y + (NOSE_LENGTH + radius) * angle.sin();
    draw_line(pos.x, pos.y, end_x, end_y, 1.0, color);
}

pub fn draw_predator(pos: Vec2, angle: f32, draw_sight_lines: bool) {
    draw_circle(pos.x, pos.y, settings::PRED_RADIUS, settings::PRED_COLOR);
    draw_nose(pos, angle, settings::PRED_RADIUS, settings::PRED_COLOR);
    if draw_sight_lines {
        draw_predator_sight(pos, angle);
    }
}

pub fn draw_prey_sight(pos: Vec2, angle: f32) {
    let n = settings::PREY_SIGHT_COUNT.max(1);
    let fov_rad = settings::PREY_SIGHT_FOV.to_radians();
    let step = if n > 1 { fov_rad / (n as f32) } else { 0.0 };
    let world_w = settings::SCREEN_WIDTH as f32;
    let world_h = settings::SCREEN_HEIGHT as f32;

    for i in 0..n {
        let sight_angle = angle + step * (i as f32);

        let end_x = pos.x + settings::PREY_SIGHT_RANGE * sight_angle.cos();
        let end_y = pos.y + settings::PREY_SIGHT_RANGE * sight_angle.sin();

        draw_wrapped_line(pos, vec2(end_x, end_y), world_w, world_h, 1.0, SKYBLUE);
    }
}

pub fn draw_prey(pos: Vec2, angle: f32, draw_sight_lines: bool) {
    draw_circle(pos.x, pos.y, settings::PREY_RADIUS, settings::PREY_COLOR);
    draw_nose(pos, angle, settings::PREY_RADIUS, settings::PREY_COLOR);
    if draw_sight_lines {
        draw_prey_sight(pos, angle);
    }
}

fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color::from_rgba(r, g, b, 255)
}

pub fn draw_neural_network(nn: &NeuralNetwork, x: f32, y: f32, width: f32, height: f32) {
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
