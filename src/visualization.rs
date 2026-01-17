// src/visualization.rs
//
// Visualization utilities for rendering the simulation

use crate::data_manager::{AnimalType, Frame};
use crate::settings::{self, *};
use macroquad::prelude::*;

pub const STATS_PANEL_WIDTH: i32 = 200;

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
