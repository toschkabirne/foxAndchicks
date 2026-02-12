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

// ----------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------
//                      PARAMETERs RELEVANT FOR CHANGES
// ----------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------
//          GAME SETTINGS
// ----------------------------------------

// Mutation / Bias params, only change the numbers, not the variables
// Adjusted to match YouTube implementation:
// - new_node_proba: 0.05 (5% chance to add node)
// - new_conn_proba: 0.7 (70% chance to add connection)
// - offset_weight_proba: 0.5 (50% chance to mutate weight)
// - offset_bias_proba: 0.2 (20% chance to mutate bias)
static ADD_NEURON: RwLock<f32> = RwLock::new(0.07);
static ADD_WEIGHT: RwLock<f32> = RwLock::new(0.5);
static CHANGE_WEIGHT: RwLock<f32> = RwLock::new(0.68);
static BIAS: RwLock<f32> = RwLock::new(0.5);

// This is the step size applied when a weight is changed
pub const MUT_CHANGE_STEP: f32 = 0.05;
// Initial mutations: YouTube uses 10, giving networks more initial structure
static PREY_INIT_MUT: RwLock<usize> = RwLock::new(15);
static PRED_INIT_MUT: RwLock<usize> = RwLock::new(15);

pub const SEED: u64 = 61;

pub const PRED_INIT_NUMB: usize = 110;
pub const PREY_INIT_NUMB: usize = 450;

pub const MAX_PRED_COUNT: usize = 175;
pub const MAX_PREY_COUNT: usize = 800;

pub const PRED_SIGHT_RANGE: f32 = 500.0;
pub const PREY_SIGHT_RANGE: f32 = 250.0;

pub const MAX_TURN_ANGLE: f32 = std::f32::consts::FRAC_PI_2;

pub const SCREEN_WIDTH: i32 = 1000;
pub const SCREEN_HEIGHT: i32 = 1000;

// FRAMES_PER_SECOND = 1 virtuell second
pub const FRAMES_PER_SECOND: i32 = 30;

// ----------------------------------------
//          PREDATOR SPCIFIC parameters
// ----------------------------------------

// This is the theoretical time, how long it needs to die (IN SECONDS)
const PRED_LIFESPAN_REST: f32 = 30.0;
const PRED_LIFESPAN_SPRINT: f32 = 20.0;

// Energy gain per eaten Prey, only change the number, not Pred_energy
pub const PRED_ENERGY_GAIN: f32 = PRED_ENERGY * 0.4;

// number of frames until reproduction cooldown is over
pub const REPRO_COOLDOWN_FRAMES: i32 = 1 * FRAMES_PER_SECOND;

// This is the theoretical time, how long it needs to cross the screen width
pub const PRED_TIME_MOVE_DIST_WIDTH: f32 = 15.0;

// ----------------------------------------
//          PREY parameters
// ----------------------------------------

// gains 0.25% of its energy per full virtuell rest second
const PREY_ENERGY_GAIN_PER_REST_SEC: f32 = 1.0;
// seconds until birth
const PREY_SECONDS_UNTIL_BIRTH: f32 = 14.0;

// seconds until 0 energy in full sprint
const PREY_LIFESPAN_SPRINT: f32 = 20.0;

// ----------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------
//                      PARAMETERS SEMI-RELEVANT FOR CHANGES
// ----------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------

// ----------------------------------------
//          GAME SETTINGS
// ----------------------------------------

pub const PREY_SIGHT_COUNT: usize = 36;
pub const PRED_SIGHT_COUNT: usize = 24;

pub const PREY_SIGHT_ANGLE: f32 = 320.0;
pub const PRED_SIGHT_ANGLE: f32 = 60.0;

pub const PRED_RADIUS: f32 = 5.0;
pub const PREY_RADIUS: f32 = 3.0;

pub const DEFAULT_TOTAL_FRAMES: i32 = 40000;

// REAC = False, not in use, but could later be used for changing acitvation functions in NN
pub const REAC: bool = false;

///Not Really Relevant for Changes
pub const PRED_COLOR: Color = RED;
pub const PREY_COLOR: Color = BLUE;
pub const BACKGROUND_COLOR: Color = BLACK;
pub const DEFAULT_DATA_FILE: &str = "simulation_data.bin";

pub const PRED_ENERGY: f32 = 100.0;
pub const PREY_ENERGY: f32 = 100.0;

pub const PREY_SPEED: f32 = 0.9 * PRED_SPEED;

// ----------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------
//                      PARAMETERS NOT TO BE CHANGED
// ----------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------

// ----------------------------------------
//          PREDATOR SPECIFIC parameters
// ----------------------------------------

// only change PRED_TIME_MOVE_DIST_WIDTH
pub const PRED_SPEED: f32 =
    (SCREEN_WIDTH as f32) / ((FRAMES_PER_SECOND as f32) * PRED_TIME_MOVE_DIST_WIDTH);
// only change PRED_LIFESPAN_REST
pub const PRED_DEFAULT_DECAY: f32 = PRED_ENERGY / (PRED_LIFESPAN_REST * (FRAMES_PER_SECOND as f32));
// only change PRED_LIFESPAN_SPRINT
pub const PRED_MOVING_DECAY: f32 = {
    let sprint_total = PRED_ENERGY / (PRED_LIFESPAN_SPRINT * FRAMES_PER_SECOND as f32);
    let extra = sprint_total - PRED_DEFAULT_DECAY;
    if extra > 0.0 {
        extra
    } else {
        0.0
    }
};

// ----------------------------------------
//          PREY SPECIFIC parameters
// ----------------------------------------

// decay resulting from sprinting, only change PREY_LIFESPAN_SPRINT
pub const PREY_MOVING_DECAY: f32 =
    PREY_ENERGY / (PREY_LIFESPAN_SPRINT * (FRAMES_PER_SECOND as f32));
// energy gain per second while resting, only change PREY_ENERGY_GAIN_PER_REST_SEC
pub const PREY_REST_ENERGY_GAIN: f32 =
    PREY_ENERGY / (FRAMES_PER_SECOND as f32) * PREY_ENERGY_GAIN_PER_REST_SEC;
// reproduction rate, only change PREY_SECONDS_UNTIL_BIRTH
pub const PREY_REPRODUCATION_RATE: f32 = PREY_SECONDS_UNTIL_BIRTH * FRAMES_PER_SECOND as f32;

// ------------------------------------------------------------
//                      HELPER FUNCTIONS FOR MUTATIONSSTUFF
// ------------------------------------------------------------

pub fn prey_init_mut() -> usize {
    *PREY_INIT_MUT.read().unwrap()
}
pub fn set_prey_init_mut(v: usize) {
    *PREY_INIT_MUT.write().unwrap() = v;
}

pub fn pred_init_mut() -> usize {
    *PRED_INIT_MUT.read().unwrap()
}
pub fn set_pred_init_mut(v: usize) {
    *PRED_INIT_MUT.write().unwrap() = v;
}

pub fn add_neuron() -> f32 {
    *ADD_NEURON.read().unwrap()
}
pub fn set_add_neuron(v: f32) {
    *ADD_NEURON.write().unwrap() = v;
}

pub fn add_weight() -> f32 {
    *ADD_WEIGHT.read().unwrap()
}
pub fn set_add_weight(v: f32) {
    *ADD_WEIGHT.write().unwrap() = v;
}

pub fn change_weight() -> f32 {
    *CHANGE_WEIGHT.read().unwrap()
}
pub fn set_change_weight(v: f32) {
    *CHANGE_WEIGHT.write().unwrap() = v;
}

pub fn bias() -> f32 {
    *BIAS.read().unwrap()
}
pub fn set_bias(v: f32) {
    *BIAS.write().unwrap() = v;
}
