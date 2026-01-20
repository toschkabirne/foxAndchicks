// src/settings.rs
//
// Auto-translated from settings.py (keep formulas identical).
// Note: In Rust we use explicit numeric types.
// - Probabilities / ratios -> f32
// - Sizes / counts -> i32 or usize
// - Energies / speeds / decays -> f32

// --------------------
use macroquad::prelude::Color;
use macroquad::prelude::{BLACK, BLUE, RED};
use std::sync::RwLock;

// Python: REAC = False (currently unused in your posted code)
pub const REAC: bool = false;

// --------------------
// game.rs settings
// --------------------
pub const BACKGROUND_COLOR: Color = BLACK;
pub const SCREEN_WIDTH: i32 = 1000;
pub const SCREEN_HEIGHT: i32 = 1000;
pub const FRAMES_PER_SECOND: i32 = 30;
pub const DEFAULT_TOTAL_FRAMES: i32 = 2000;

pub const PRED_INIT_NUMB: usize = 100;
pub const PREY_INIT_NUMB: usize = 200;

pub const MAX_PRED_COUNT: usize = 600;
pub const MAX_PREY_COUNT: usize = 1200;

// --------------------
// data_manager.rs settings: #Rendering Engine
// --------------------
pub const SIGHT_RANGE_PREDATOR: f32 = 150.0;
pub const SIGHT_RANGE_PREY: f32 = 100.0;

pub const NUMBER_SIGHTS_PREY: usize = 24;
pub const NUMBER_SIGHTS_PREDATOR: usize = 24;

pub const PREDATOR_COLOR: Color = RED;
pub const PREY_COLOR: Color = BLUE;

// --------------------
// Predator movement / energy
// --------------------

pub const PREDATOR_RADIUS: f32 = 10.0;
pub const PREY_RADIUS: f32 = 7.0;

pub const PRED_TIME_MOVE_DIST_WIDTH: f32 = 15.0;

// PREDATOR_SPEED = SCREEN_WIDTH/(FRAMES_PER_SECOND*PRED_TIME_MOVE_DIST_WIDTH)
pub const PREDATOR_SPEED: f32 =
    (SCREEN_WIDTH as f32) / ((FRAMES_PER_SECOND as f32) * PRED_TIME_MOVE_DIST_WIDTH);

// Energy
pub const PRED_ENERGY: f32 = 100.0;
pub const PREDATOR_ENERGY_GAIN: f32 = 40.0;
pub const PREDATOR_LIFESPAN: f32 = 40.0;

// PRED_DEFAULT_DECAY = PRED_ENERGY / (PREDATOR_LIFESPAN*FRAMES_PER_SECOND)
pub const PRED_DEFAULT_DECAY: f32 = PRED_ENERGY / (PREDATOR_LIFESPAN * (FRAMES_PER_SECOND as f32));

// PRED_MOVING_DECAY = PRED_ENERGY/(2*SCREEN_WIDTH)
pub const PRED_MOVING_DECAY: f32 = PRED_ENERGY / (2.0 * (SCREEN_WIDTH as f32));

// --------------------
// Prey movement / energy
// --------------------

// Python:
// PREY_SPEED = 0.8 * PREDATOR_SPEED
pub const PREY_SPEED: f32 = 0.8 * PREDATOR_SPEED;

// Energy
pub const PREY_ENERGY: f32 = 50.0;
pub const PREY_REPRODUCATION_RATE: f32 = 16.0;

// Python:
// PREY_MOVING_DECAY = PRED_ENERGY/(0.2*SCREEN_WIDTH)
pub const PREY_MOVING_DECAY: f32 = PRED_ENERGY / (0.2 * (SCREEN_WIDTH as f32));

// Python:
// PREY_REST_ENERGY_GAIN = PREY_ENERGY/(4*FRAMES_PER_SECOND)
pub const PREY_REST_ENERGY_GAIN: f32 = PREY_ENERGY / (4.0 * (FRAMES_PER_SECOND as f32));

// --------------------
// Mutations parameter
// --------------------
static PREY_INIT_MUT: RwLock<usize> = RwLock::new(40);
pub fn prey_init_mut() -> usize {
    *PREY_INIT_MUT.read().unwrap()
}
pub fn set_prey_init_mut(v: usize) {
    *PREY_INIT_MUT.write().unwrap() = v;
}

static PRED_INIT_MUT: RwLock<usize> = RwLock::new(40);
pub fn pred_init_mut() -> usize {
    *PRED_INIT_MUT.read().unwrap()
}
pub fn set_pred_init_mut(v: usize) {
    *PRED_INIT_MUT.write().unwrap() = v;
}

pub const DEFAULT_DATA_FILE: &str = "simulation_data.bin";

// --------------------
// Mutation / Bias params
// --------------------
static ADD_NEURON: RwLock<f32> = RwLock::new(0.07);
pub fn add_neuron() -> f32 {
    *ADD_NEURON.read().unwrap()
}
pub fn set_add_neuron(v: f32) {
    *ADD_NEURON.write().unwrap() = v;
}

static ADD_WEIGHT: RwLock<f32> = RwLock::new(0.65);
pub fn add_weight() -> f32 {
    *ADD_WEIGHT.read().unwrap()
}
pub fn set_add_weight(v: f32) {
    *ADD_WEIGHT.write().unwrap() = v;
}

static CHANGE_WEIGHT: RwLock<f32> = RwLock::new(0.9);
pub fn change_weight() -> f32 {
    *CHANGE_WEIGHT.read().unwrap()
}
pub fn set_change_weight(v: f32) {
    *CHANGE_WEIGHT.write().unwrap() = v;
}

static BIAS: RwLock<f32> = RwLock::new(0.5);
pub fn bias() -> f32 {
    *BIAS.read().unwrap()
}
pub fn set_bias(v: f32) {
    *BIAS.write().unwrap() = v;
}
