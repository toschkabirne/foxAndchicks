// src/visualization.rs
//
// Visualization utilities for rendering the simulation

use crate::data_manager::{AnimalType, Frame};
use crate::settings::{self, *};
use macroquad::prelude::*;

pub const STATS_PANEL_WIDTH: i32 = 200;
pub const PLAYBACK_CONTROLS_HEIGHT: f32 = 60.0;

/// Returns the window configuration for macroquad
pub fn window_conf() -> Conf {
    Conf {
        window_title: "Predator and Prey Simulation".to_string(),
        window_width: settings::SCREEN_WIDTH + STATS_PANEL_WIDTH,
        window_height: settings::SCREEN_HEIGHT,
        ..Default::default()
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

    draw_text(&text_pred, text_x, 70.0, 20.0, settings::PREDATOR_COLOR);
    draw_text(&text_prey, text_x, 95.0, 20.0, settings::PREY_COLOR);
    draw_text(&frame_text, text_x, 130.0, 20.0, WHITE);
}

/// Draws all animals in the given frame
pub fn draw_frame(frame: &Frame, draw_sight_lines: bool) {
    for animal in &frame.animals {
        match animal.animal_type {
            AnimalType::Predator => {
                // Draw sight lines for predator
                let start_angle = animal.angle - 30.0_f32.to_radians();
                let end_angle = animal.angle + 30.0_f32.to_radians();

                if draw_sight_lines {
                    for i in 0..NUMBER_SIGHTS_PREDATOR {
                        let t = if NUMBER_SIGHTS_PREDATOR > 1 {
                            i as f32 / (NUMBER_SIGHTS_PREDATOR as f32 - 1.0)
                        } else {
                            0.0
                        };
                        let sight_angle = start_angle + t * (end_angle - start_angle);

                        let end_x = animal.x + SIGHT_RANGE_PREDATOR * sight_angle.cos();
                        let end_y = animal.y + SIGHT_RANGE_PREDATOR * sight_angle.sin();

                        draw_line(animal.x, animal.y, end_x, end_y, 1.0, YELLOW);
                    }
                }
                draw_circle(animal.x, animal.y, PREDATOR_RADIUS, PREDATOR_COLOR);
            }
            AnimalType::Prey => {
                // Draw sight lines for prey
                if draw_sight_lines {
                    for i in 0..NUMBER_SIGHTS_PREY {
                        let sight_angle = animal.angle
                            + (360.0 / NUMBER_SIGHTS_PREY as f32).to_radians() * i as f32;

                        let end_x = animal.x + SIGHT_RANGE_PREY * sight_angle.cos();
                        let end_y = animal.y + SIGHT_RANGE_PREY * sight_angle.sin();

                        draw_line(animal.x, animal.y, end_x, end_y, 1.0, SKYBLUE);
                    }
                }
                draw_circle(animal.x, animal.y, PREY_RADIUS, PREY_COLOR);
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
pub fn draw_playback_controls(
    state: &mut PlaybackState,
    total_frames: usize,
) {
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
        draw_rectangle(
            button_x + 8.0,
            bar_y,
            bar_width,
            bar_height,
            WHITE,
        );
        draw_rectangle(
            button_x + 16.0,
            bar_y,
            bar_width,
            bar_height,
            WHITE,
        );
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
