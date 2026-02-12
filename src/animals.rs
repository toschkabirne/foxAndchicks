// ============================================================================
// Dependencies
// ============================================================================
use crate::brain_neural_network::NeuralNetwork;
use crate::settings::*;
use crate::spatial_hash::HasPos;

use ::rand::Rng;
use macroquad::prelude::*;
use std::iter::IntoIterator;

use std::collections::HashSet;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};

// ============================================================================
// GLOBAL STATE AND CONSTANTS
// ============================================================================

const THRESHOLD_PRED: f32 = 0.05;
const THRESHOLD_PREY: f32 = 0.06;
/// Global counter for assigning unique IDs to animals, needed for tracking during hunting, reproduction, and data collection
static NEXT_ID: AtomicUsize = AtomicUsize::new(1); // thread-safe ID generation

/// Pre-calculated constant for 2π (full circle in radians).
const TWO_PI: f32 = 2.0 * PI;

// helper function to generate unique IDs for animals
#[inline]
fn next_id() -> usize {
    assert!(NEXT_ID.load(Ordering::Relaxed) > 0);
    assert!(NEXT_ID.load(Ordering::Relaxed) < usize::MAX);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

// ============================================================================
// HELPER FUNCTIONS - TOROIDAL WORLD GEOMETRY
// ============================================================================
// We implement a "wrap-around" world where entities crossing one edge appear
// on the opposite edge. This eliminates edge effects where animals could get
// "stuck" in corners, creating a more uniform simulation environment. This
// is especially important for evolution simulations where environmental biases
// should be minimized.
// ============================================================================

/// Wraps a position to stay within world boundaries (toroidal topology).
#[inline]
pub fn wrap_position(pos: Vec2, width: f32, height: f32) -> Vec2 {
    assert!(width > 0.0 && height > 0.0);
    vec2(pos.x.rem_euclid(width), pos.y.rem_euclid(height))
}

/// Computes the shortest vector from point `a` to point `b` in a toroidal world.
#[inline]
pub fn wrapped_distance_vector(a: Vec2, b: Vec2, width: f32, height: f32) -> Vec2 {
    assert!(width > 0.0 && height > 0.0);
    let mut dx = b.x - a.x;
    let mut dy = b.y - a.y;

    // If the direct distance is more than half the world width,
    // the wrapped distance (going the other way) is shorter
    if dx > width / 2.0 {
        dx -= width; // Wrap left
    } else if dx < -width / 2.0 {
        dx += width; // Wrap right
    }
    // Same logic for vertical distance
    if dy > height / 2.0 {
        dy -= height; // Wrap up
    } else if dy < -height / 2.0 {
        dy += height; // Wrap down
    }
    assert!(dx >= -width / 2.0 && dx <= width / 2.0);
    assert!(dy >= -height / 2.0 && dy <= height / 2.0);

    vec2(dx, dy)
}

/// Computes the shortest Euclidean distance between two points in a toroidal world.
#[inline]
pub fn wrapped_distance_abs(a: Vec2, b: Vec2, width: f32, height: f32) -> f32 {
    assert!(width > 0.0 && height > 0.0);
    wrapped_distance_vector(a, b, width, height).length()
}

/// Normalizes an angle to the range [-π, π].
#[inline]
pub fn normalize_angle(angle: f32) -> f32 {
    // Ensure the angle is within the range [-PI, PI].
    (angle + PI).rem_euclid(TWO_PI) - PI
}

// ============================================================================
// SHARED ANIMAL CORE
// ============================================================================

/// Core data shared by all animals (Predators and Prey).
///
/// This struct contains the fundamental state that every animal in the
/// simulation needs, regardless of its specific role.
#[derive(Clone)]
pub struct AnimalCore {
    /// Unique identifier for this animal (never reused, even after death)
    pub id: usize,
    /// Current 2D position in the world
    pub pos: Vec2,
    /// Current heading angle in radians (0 = facing right/east)
    pub angle: f32,
    /// Current energy level (animal dies/cannot reproduce when too low)
    pub energy: f32,
    /// Neural network brain that controls decision-making, each animal owns its brain
    pub brain: NeuralNetwork,
    pub survived_iters: i32,
}

pub trait HasCore {
    fn core(&self) -> &AnimalCore;
}

impl HasCore for Prey {
    fn core(&self) -> &AnimalCore {
        &self.core
    }
}

impl HasCore for Predator {
    fn core(&self) -> &AnimalCore {
        &self.core
    }
}

impl AnimalCore {
    /// Creates a new AnimalCore with a new brain
    pub fn new_with_brain(pos: Vec2, angle: f32, energy: f32, brain: NeuralNetwork) -> Self {
        Self {
            id: next_id(),
            pos,
            angle,
            energy,
            brain,
            survived_iters: 0,
        }
    }

    #[inline]
    pub fn x(&self) -> f32 {
        assert!(self.pos.x >= 0.0 && self.pos.x <= SCREEN_WIDTH as f32);
        self.pos.x
    }

    #[inline]
    pub fn y(&self) -> f32 {
        assert!(self.pos.y >= 0.0 && self.pos.y <= SCREEN_HEIGHT as f32);
        self.pos.y
    }

    #[inline]
    pub fn set_xy(&mut self, x: f32, y: f32) {
        assert!(x >= 0.0 && x <= SCREEN_WIDTH as f32);
        assert!(y >= 0.0 && y <= SCREEN_HEIGHT as f32);
        self.pos = vec2(x, y);
    }
}

/// Computes sensory inputs for the animal's neural network.
///
/// Animals have forward-facing "cone vision" covering ±sight_angle from their heading.
/// The cone is divided into NUMBER_SIGHTS rays, each returning a normalized distance value:
/// - 1.0 if enemy is very close (distance = 0)
/// - 0.0 if enemy is at max sight range or not detected
/// - Values in between represent normalized proximity (closer = higher value)
impl AnimalCore {
    pub fn sense_animals<'a, I, T>(
        core: &AnimalCore,
        enemies: I,
        number_sights: usize,
        sight_range: f32,
        sight_angle: f32,
        enemy_radius: f32,
    ) -> Vec<f32>
    where
        I: IntoIterator<Item = &'a T>,
        T: HasCore + 'a,
    {
        assert!(number_sights > 0);
        let n = number_sights.max(1);
        let mut inputs = vec![0.0; n];

        let cone_half_angle = (sight_angle / 2.0_f32).to_radians();
        assert!(cone_half_angle > 0.0 && cone_half_angle < PI);

        let animal_pos = core.pos;
        assert!(animal_pos.x >= 0.0 && animal_pos.x <= SCREEN_WIDTH as f32);
        assert!(animal_pos.y >= 0.0 && animal_pos.y <= SCREEN_HEIGHT as f32);
        let world_w = SCREEN_WIDTH as f32;
        let world_h = SCREEN_HEIGHT as f32;

        for enemy in enemies {
            let enemy_pos = enemy.core().pos;
            assert!(enemy_pos.x >= 0.0 && enemy_pos.x <= SCREEN_WIDTH as f32);
            assert!(enemy_pos.y >= 0.0 && enemy_pos.y <= SCREEN_HEIGHT as f32);

            // Compute shortest vector to enemy (accounting for world wrapping)
            let delta = wrapped_distance_vector(animal_pos, enemy_pos, world_w, world_h);
            let dist = delta.length();
            //
            assert!(dist >= 0.0);

            // Skip if enemy is too close (div-by-zero) or out of range
            if dist == 0.0 || dist >= sight_range {
                continue;
            }
            // Compute angle to enemy using wrapped delta
            let angle_to_enemy = delta.y.atan2(delta.x);
            assert!((-PI..=PI).contains(&angle_to_enemy));

            let angular_width = (enemy_radius).atan2(dist);
            // Normalize distance: 1.0 = very close, 0.0 = at max range
            let normalized_proximity = 1.0 - (dist / sight_range);
            assert!((0.0..=1.0).contains(&normalized_proximity));

            // Check each vision ray to see if enemy overlaps with it
            for (i, inp) in inputs.iter_mut().enumerate().take(n) {
                // Interpolate angle for this ray within the vision cone
                let t = if n > 1 {
                    i as f32 / (n as f32 - 1.0) // Map index to [0, 1]
                } else {
                    0.5 // Single ray: use middle of cone
                };
                assert!((0.0..=1.0).contains(&t));

                let offset = -cone_half_angle + (2.0 * cone_half_angle) * t; // [-cone_half .. +cone_half]
                let ray_angle = normalize_angle(core.angle + offset);
                assert!((-PI..=PI).contains(&ray_angle));

                // Check if enemy's angular disc overlaps this ray
                let diff = normalize_angle(ray_angle - angle_to_enemy).abs();
                if diff <= angular_width {
                    // Store the closest enemy's normalized distance for this ray
                    if normalized_proximity > *inp {
                        *inp = normalized_proximity;
                    }
                }
            }
        }

        inputs
    }
    pub fn sense_animals_optimized<'a, I, T>(
        core: &AnimalCore,
        enemies: I,
        number_sights: usize,
        sight_range: f32,
        sight_angle: f32,
        enemy_radius: f32,
        inputs: &mut [f32],
    ) where
        I: IntoIterator<Item = &'a T>,
        T: HasCore + 'a,
    {
        debug_assert!(number_sights > 0);

        if inputs.is_empty() {
            return;
        }

        // clamp n to available buffer length
        let n = number_sights.max(1).min(inputs.len());

        for v in &mut inputs[..n] {
            *v = 0.0;
        }

        let mut remaining = n;

        let cone_half = (sight_angle * 0.5).to_radians();
        debug_assert!(cone_half.is_finite() && cone_half > 0.0 && cone_half < std::f32::consts::PI);

        let animal_pos = core.pos;
        let world_w = SCREEN_WIDTH as f32;
        let world_h = SCREEN_HEIGHT as f32;

        let step = if n > 1 {
            (2.0 * cone_half) / ((n - 1) as f32)
        } else {
            0.0
        };
        debug_assert!(n == 1 || (step.is_finite() && step > 0.0));

        let mut candidates: Vec<(f32, f32, f32)> = Vec::new();

        for enemy in enemies {
            let enemy_pos = enemy.core().pos;

            let delta = wrapped_distance_vector(animal_pos, enemy_pos, world_w, world_h);
            let dist = delta.length();

            if dist == 0.0 || dist >= sight_range {
                continue;
            }

            let angle_to_enemy = delta.y.atan2(delta.x);
            let center_offset = normalize_angle(angle_to_enemy - core.angle);
            let angular_width = enemy_radius.atan2(dist);

            if center_offset.abs() > (cone_half + angular_width) {
                continue;
            }

            candidates.push((dist, center_offset, angular_width));
        }

        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for (dist, center_offset, angular_width) in candidates {
            if remaining == 0 {
                break;
            }

            // Normalize distance: 1.0 = very close, 0.0 = at max range
            let normalized_proximity = 1.0 - (dist / sight_range);
            debug_assert!((0.0..=1.0).contains(&normalized_proximity));

            let mut start = center_offset - angular_width;
            let mut end = center_offset + angular_width;

            if start < -cone_half {
                start = -cone_half;
            }
            if end > cone_half {
                end = cone_half;
            }

            if start > end {
                continue;
            }

            if n == 1 {
                if start <= 0.0 && 0.0 <= end && normalized_proximity > inputs[0] {
                    inputs[0] = normalized_proximity;
                    if inputs[0] >= 1.0 {
                        remaining = 0;
                    }
                }
                continue;
            }

            let i_min_f = (start + cone_half) / step;
            let i_max_f = (end + cone_half) / step;

            if !i_min_f.is_finite() || !i_max_f.is_finite() {
                continue;
            }

            let mut i_min = i_min_f.ceil() as isize;
            let mut i_max = i_max_f.floor() as isize;

            if i_min < 0 {
                i_min = 0;
            }
            if i_max > (n as isize - 1) {
                i_max = n as isize - 1;
            }

            if i_min > i_max {
                continue;
            }

            for inp in inputs
                .iter_mut()
                .take(i_max as usize + 1)
                .skip(i_min as usize)
            {
                // Store the closest enemy's normalized distance for this ray
                if normalized_proximity > *inp {
                    *inp = normalized_proximity;
                    if *inp >= 1.0 {
                        remaining -= 1;
                        if remaining == 0 {
                            break;
                        }
                    }
                }
            }
        }
    }
}

/// Creates a child brain from a parent brain with random mutations.
#[inline]
fn inherited_brain_with_mutations<R: Rng>(parent: &NeuralNetwork, rng: &mut R) -> NeuralNetwork {
    let mut brain = parent.clone();
    let k = rng.gen_range(2..=6); // Apply multiple mutations
    for _ in 0..k {
        brain.mutate(rng);
    }
    brain
}

/// Performs a movement step for an animal, updating position, angle, and energy.
/// This is shared movement logic used by both Predators and Prey.
/// 1. **Quadratic energy cost**: Energy cost is proportional to speed_factor²,
/// 2. **Energy capping**: We prevent energy from exceeding max_energy
/// 3. **Separated turning and movement**: Animals first turn, then move forward
#[inline]
fn move_with_speed_factor(
    core: &mut AnimalCore,
    speed_factor: f32, // change in speed (negative = backwards)
    turn_delta: f32,   // change in angle
    speed: f32,        // base speed
    moving_decay: f32, // energy cost multiplier for movement
) {
    assert!((-1.0..=1.0).contains(&speed_factor));
    assert!((-PI..=PI).contains(&turn_delta));
    assert!((0.0..=1.0).contains(&moving_decay));
    // Apply turning
    core.angle = normalize_angle(core.angle + turn_delta);
    assert!(core.angle >= -PI && core.angle <= PI);

    // Apply forward/backward movement based on current heading
    // Negative speed_factor moves backwards
    core.pos.x += speed_factor * speed * core.angle.cos();
    core.pos.y += speed_factor * speed * core.angle.sin();

    // Handle world wrapping (toroidal topology)
    core.pos = wrap_position(core.pos, SCREEN_WIDTH as f32, SCREEN_HEIGHT as f32);

    assert!(core.pos.x >= 0.0 && core.pos.x <= SCREEN_WIDTH as f32);
    assert!(core.pos.y >= 0.0 && core.pos.y <= SCREEN_HEIGHT as f32);
    // Deduct energy cost (quadratic in speed for realism)
    // Use abs() so backwards movement also costs energy
    let v = speed_factor.abs();
    core.energy -= PRED_MOVING_DECAY * v.sqrt();
}

// ============================================================================
// PREDATOR IMPLEMENTATION
// ============================================================================
// Design rationale for Predator:
// Predators are the "hunters" in the simulation. They must:
// 1. Detect prey using limited forward-facing vision
// 2. Chase and catch prey (collision detection)
// 3. Manage energy carefully (hunting costs energy, eating restores it)
// 4. Reproduce when successful (evolutionary reward for good hunters)
//
// The selective pressure on predators:
// - Must balance aggressive hunting (high energy cost) vs. conservation
// - Must develop effective vision-based tracking behaviors
// - Population naturally limits itself (no prey = predators starve)
// ============================================================================

/// A predator animal that hunts prey.
#[derive(Clone)]
pub struct Predator {
    /// Core animal state (position, angle, energy, brain)
    pub core: AnimalCore,

    /// Counter for prey eaten since last reproduction, if >c they reproduce
    pub eaten_prey: i32,

    /// Lifetime kill count (never reset, used for top predator selection)
    pub lifetime_kills: i32,

    /// Cooldown timer preventing immediate re-reproduction (applies per frame)
    pub repro_cooldown: i32,
}

/// Trait implementation for spatial hash queries.
impl HasPos for Predator {
    fn x(&self) -> f32 {
        assert!(self.core.x() >= 0.0 && self.core.x() <= SCREEN_WIDTH as f32);
        self.core.x()
    }
    fn y(&self) -> f32 {
        assert!(self.core.y() >= 0.0 && self.core.y() <= SCREEN_HEIGHT as f32);
        self.core.y()
    }
    fn set_pos(&mut self, x: f32, y: f32) {
        assert!(x >= 0.0 && x <= SCREEN_WIDTH as f32);
        assert!(y >= 0.0 && y <= SCREEN_HEIGHT as f32);
        self.core.set_xy(x, y);
    }
}

impl Predator {
    /// Creates a new predator with a random brain and position.
    pub fn new<R: Rng>(x: f32, y: f32, rng: &mut R) -> Self {
        assert!(x >= 0.0 && x <= SCREEN_WIDTH as f32);
        assert!(y >= 0.0 && y <= SCREEN_HEIGHT as f32);
        let angle = rng.gen_range(0.0..TWO_PI);
        // Create brain with predator-specific input count and mutation rate
        let brain = NeuralNetwork::new(PRED_SIGHT_COUNT, 2, pred_init_mut(), bias(), rng);
        assert!(brain.num_inputs == PRED_SIGHT_COUNT);
        assert!(brain.num_outputs == 2);
        Self {
            core: AnimalCore::new_with_brain(vec2(x, y), angle, PRED_ENERGY, brain),
            eaten_prey: 0,
            lifetime_kills: 0,
            repro_cooldown: 0,
        }
    }

    /// Returns the unique ID of this predator.
    #[inline]
    pub fn id(&self) -> usize {
        self.core.id
    }

    /// Executes one movement/decision step for the predator.
    ///
    /// Design rationale:
    /// 1. Predators lose energy each frame just for existing (PRED_DEFAULT_DECAY)
    /// 2. Energy ratio input: The brain receives energy_ratio (current/max)
    ///    as a bias input. This allows the brain to "know" it's own energy state
    pub fn move_step(&mut self, inputs: &[f32]) {
        // Passive energy decay (cost of living)
        self.core.energy -= PRED_DEFAULT_DECAY;

        if self.core.energy <= 0.0 {
            return;
        }

        assert!(self.core.energy >= 0.0 && self.core.energy <= PRED_ENERGY);

        // Compute energy ratio for brain decision-making
        let energy_ratio = self.core.energy / PRED_ENERGY;

        assert!((0.0..=1.0).contains(&energy_ratio));

        // Run neural network to get movement decisions
        let outputs = self.core.brain.forward_vectorized(inputs, energy_ratio);

        // Extract and clamp movement parameters
        // Allow backwards movement: -1.0 = full reverse, 1.0 = full forward
        let speed_factor = outputs[0].clamp(-1.0, 1.0);
        let turn_delta = outputs[1].clamp(-1.0, 1.0) * MAX_TURN_ANGLE;
        // threshold for moving, and if low on energy, stop moving, gain energy
        if speed_factor < THRESHOLD_PRED {
            self.core.survived_iters += 1;
            return; // Do not move if below threshold
        }

        // Apply movement (includes additional energy cost)
        move_with_speed_factor(
            &mut self.core,
            speed_factor,
            turn_delta,
            PRED_SPEED,
            PRED_MOVING_DECAY,
        );
        self.core.survived_iters += 1;
    }

    /// Checks for nearby prey and attempts to eat them
    pub fn hunt_nearby<'a, R: Rng, I>(
        &mut self,
        prey_candidates: I,
        eaten_prey_ids: &mut HashSet<usize>,
        newborn_preds: &mut Vec<Predator>,
        rng: &mut R,
    ) where
        I: IntoIterator<Item = &'a Prey>,
    {
        // Collision threshold: predator and prey radii combined
        let eat_r = PRED_RADIUS + PREY_RADIUS;

        assert!(eat_r >= 0.0);

        let predator_pos = self.core.pos;
        let world_w = SCREEN_WIDTH as f32;
        let world_h = SCREEN_HEIGHT as f32;

        for prey in prey_candidates {
            let id = prey.core.id;

            // Skip prey that have already been eaten this frame
            if eaten_prey_ids.contains(&id) {
                continue;
            }

            // Check if close enough to eat (using toroidal distance)
            let prey_pos = prey.core.pos;
            let dist = wrapped_distance_abs(predator_pos, prey_pos, world_w, world_h);

            assert!(dist >= 0.0);

            if dist < eat_r {
                // Mark prey as eaten (prevents double-eating)
                eaten_prey_ids.insert(id);

                // Gain energy (capped at maximum)
                self.core.energy = (self.core.energy + PRED_ENERGY_GAIN).min(PRED_ENERGY);

                // Increment kill counters
                self.eaten_prey += 1;
                self.lifetime_kills += 1;

                // Attempt to reproduce (may succeed if threshold reached)
                if let Some(child) = self.reproduce(rng) {
                    newborn_preds.push(child);
                }

                // Allows only one kill per frame
                break;
            }
        }
    }

    /// Attempts to reproduce, creating an offspring if conditions are met.
    pub fn reproduce<R: Rng>(&mut self, rng: &mut R) -> Option<Predator> {
        // Cooldown timer preventing immediate re-reproduction (applies per frame)

        const { assert!(REPRO_COOLDOWN_FRAMES > 0) };

        // Check cooldown
        if self.repro_cooldown > 0 {
            return None;
        }

        assert!(self.repro_cooldown >= 0);

        // Check kill threshold
        if self.eaten_prey < 3 {
            return None;
        }

        assert!(self.eaten_prey >= 0);

        // Reset counters for next reproduction cycle
        self.eaten_prey = 0;
        self.repro_cooldown = REPRO_COOLDOWN_FRAMES;

        // Tiny offset near parent
        // let ox = self.core.pos.x + rng.gen_range(-1..=1) as f32;
        // let oy = self.core.pos.y + rng.gen_range(-1..=1) as f32;

        // // Child gets parent's brain + mutations
        // let mut child = Predator::new(ox, oy, rng);

        // Calculate position with wrap-around handling

        let pos = wrap_position(
            vec2(
                self.core.pos.x + rng.gen_range(-1..=1) as f32,
                self.core.pos.y + rng.gen_range(-1..=1) as f32,
            ),
            SCREEN_WIDTH as f32,
            SCREEN_HEIGHT as f32,
        );

        assert!(pos.x >= 0.0 && pos.x <= SCREEN_WIDTH as f32);
        assert!(pos.y >= 0.0 && pos.y <= SCREEN_HEIGHT as f32);

        let mut child = Predator::new(pos.x, pos.y, rng);

        child.core.brain = inherited_brain_with_mutations(&self.core.brain, rng);

        // Child starts with full energy (not split from parent)
        child.core.energy = PRED_ENERGY;

        Some(child)
    }
}

impl Predator {
    pub fn get_inputs<'a, I>(&self, preys: I) -> Vec<f32>
    where
        I: IntoIterator<Item = &'a Prey>,
    {
        AnimalCore::sense_animals(
            &self.core,
            preys,
            PRED_SIGHT_COUNT,
            PRED_SIGHT_RANGE,
            PRED_SIGHT_ANGLE,
            PREY_RADIUS,
        )
    }

    pub fn get_inputs_optimized<'a, I>(&self, preys: I, inputs: &mut [f32])
    where
        I: IntoIterator<Item = &'a Prey>,
    {
        AnimalCore::sense_animals_optimized(
            &self.core,
            preys,
            PRED_SIGHT_COUNT,
            PRED_SIGHT_RANGE,
            PRED_SIGHT_ANGLE,
            PREY_RADIUS,
            inputs,
        )
    }
}
impl Prey {
    pub fn get_inputs<'a, I>(&self, predators: I) -> Vec<f32>
    where
        I: IntoIterator<Item = &'a Predator>,
    {
        AnimalCore::sense_animals(
            &self.core,
            predators,
            PREY_SIGHT_COUNT,
            PREY_SIGHT_RANGE,
            PREY_SIGHT_ANGLE,
            PRED_RADIUS,
        )
    }
}
// ============================================================================
// PREY IMPLEMENTATION
// ============================================================================
// Design rationale for Prey:
// Prey are the "hunted" in the simulation. They must:
// 1. Detect predators using 300 vision
// 2. Evade predators through movement (run away)
// 3. Manage energy carefully (fleeing costs energy, resting recovers it)
// 4. Reproduce over time (population growth)
// - **Reproduction**: Prey reproduce on a timer (vs. predators' kill-threshold)
// - **Energy recovery**: Prey can rest to recover energy (predators cannot)
// ============================================================================

/// A prey animal that tries to avoid predators.
#[derive(Clone)]
pub struct Prey {
    /// Core animal state (position, angle, energy, brain)
    pub core: AnimalCore,

    /// This counter increments each frame and reproduction happens when it
    /// reaches a threshold (PREY_REPRODUCATION_RATE * FPS).
    pub rest_time: i32,
}

/// Trait implementation for spatial hash queries (same as Predator).
impl HasPos for Prey {
    fn x(&self) -> f32 {
        assert!(self.core.x() >= 0.0 && self.core.x() <= SCREEN_WIDTH as f32);
        self.core.x()
    }
    fn y(&self) -> f32 {
        assert!(self.core.y() >= 0.0 && self.core.y() <= SCREEN_HEIGHT as f32);
        self.core.y()
    }
    fn set_pos(&mut self, x: f32, y: f32) {
        assert!(x >= 0.0 && x <= SCREEN_WIDTH as f32);
        assert!(y >= 0.0 && y <= SCREEN_HEIGHT as f32);
        self.core.set_xy(x, y);
    }
}
impl Prey {
    /// Creates a new prey with a random brain and position.
    pub fn new<R: Rng>(x: f32, y: f32, rng: &mut R) -> Self {
        assert!(x >= 0.0 && x <= SCREEN_WIDTH as f32);
        assert!(y >= 0.0 && y <= SCREEN_HEIGHT as f32);

        let angle = rng.gen_range(0.0..TWO_PI);
        let brain = NeuralNetwork::new(PREY_SIGHT_COUNT, 2, prey_init_mut(), bias(), rng);

        assert!((0.0..TWO_PI).contains(&angle));

        Self {
            core: AnimalCore::new_with_brain(vec2(x, y), angle, PREY_ENERGY, brain),
            rest_time: 0,
        }
    }

    /// Returns the unique ID of this prey.
    #[inline]
    pub fn id(&self) -> usize {
        self.core.id
    }

    /// Executes one movement/decision step for the prey.
    /// 1. Prey don't lose energy just for existing.
    /// 2. If speed factor is below threshold, or low on energy, rest & gain energy.
    pub fn move_step(&mut self, inputs: &[f32]) {
        assert!(inputs.len() == PREY_SIGHT_COUNT);

        let energy_ratio = self.core.energy / PREY_ENERGY;
        let outputs = self.core.brain.forward_vectorized(inputs, energy_ratio);

        assert!(outputs.len() == 2);
        assert!((0.0..=1.0).contains(&energy_ratio));

        let speed_factor = outputs[0].clamp(0.0, 1.0);
        if speed_factor < THRESHOLD_PREY || self.core.energy < 0.02 * PREY_ENERGY {
            // moving threshold, rests and gains energy
            self.core.survived_iters += 1;
            self.core.energy = (self.core.energy + PREY_REST_ENERGY_GAIN).min(PREY_ENERGY);
            return; // Don't move while resting
        }

        // Apply turning and movement
        let turn_delta = outputs[1].clamp(-1.0, 1.0) * MAX_TURN_ANGLE;

        assert!((-MAX_TURN_ANGLE..=MAX_TURN_ANGLE).contains(&turn_delta));

        move_with_speed_factor(
            &mut self.core,
            speed_factor,
            turn_delta,
            PREY_SPEED,
            PREY_MOVING_DECAY,
        );
        self.core.survived_iters += 1;
    }

    /// Attempts to reproduce, creating an offspring if conditions are met.
    /// 1. Prey reproduce after surviving for a certain time. The timer (rest_time) increments every
    ///    frame, and reproduction happens when it reaches: `PREY_REPRODUCATION_RATE * FRAMES_PER_SECOND`
    /// 2. Brain inheritance: Same as predators
    pub fn reproduce<R: Rng>(&mut self, rng: &mut R, has_slot: bool) -> Option<Prey> {
        // Increment timer every frame
        self.rest_time += 1;

        // Calculate reproduction threshold (time in frames)
        let threshold = PREY_REPRODUCATION_RATE as i32;

        assert!(threshold >= 0);

        // Not ready yet
        if self.rest_time < threshold {
            return None;
        }

        // Ready to reproduce, but population is at capacity
        if !has_slot {
            // Clamp at threshold rather than resetting
            self.rest_time = threshold;
            return None;
        }

        // Slot is free -> give birth!
        self.rest_time = 0; // Reset timer for next reproduction

        // Spawn child with larger offset than predators (±50 vs ±1)
        let ox =
            rng.gen_range((self.core.pos.x as i32 - 50)..=(self.core.pos.x as i32 + 50)) as f32;
        let oy =
            rng.gen_range((self.core.pos.y as i32 - 50)..=(self.core.pos.y as i32 + 50)) as f32;

        let p = wrap_position(vec2(ox, oy), SCREEN_WIDTH as f32, SCREEN_HEIGHT as f32);

        assert!(p.x >= 0.0 && p.x <= SCREEN_WIDTH as f32);
        assert!(p.y >= 0.0 && p.y <= SCREEN_HEIGHT as f32);

        let mut child = Prey::new(p.x, p.y, rng);
        child.core.brain = inherited_brain_with_mutations(&self.core.brain, rng);

        assert!(child.core.energy >= 0.0 && child.core.energy <= PREY_ENERGY);
        assert!(child.core.brain.num_inputs == PREY_SIGHT_COUNT);
        assert!(child.core.brain.num_outputs == 2);

        Some(child)
    }
}

// ============================================================================
// TEST FUNCTIONS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const WORLD_W: f32 = 1000.0;
    const WORLD_H: f32 = 1000.0;
    const EPSILON: f32 = 0.0001;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn vec_approx_eq(a: Vec2, b: Vec2) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y)
    }

    #[test]
    fn test_wrap_position_no_wrap_needed() {
        let pos = vec2(500.0, 500.0);
        let wrapped = wrap_position(pos, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(wrapped, pos));
    }

    #[test]
    fn test_wrap_position_x_overflow() {
        let pos = vec2(1050.0, 500.0);
        let wrapped = wrap_position(pos, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(wrapped, vec2(50.0, 500.0)));
    }

    #[test]
    fn test_wrap_position_x_underflow() {
        let pos = vec2(-50.0, 500.0);
        let wrapped = wrap_position(pos, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(wrapped, vec2(950.0, 500.0)));
    }

    #[test]
    fn test_wrap_position_y_overflow() {
        let pos = vec2(500.0, 1100.0);
        let wrapped = wrap_position(pos, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(wrapped, vec2(500.0, 100.0)));
    }

    #[test]
    fn test_wrap_position_y_underflow() {
        let pos = vec2(500.0, -100.0);
        let wrapped = wrap_position(pos, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(wrapped, vec2(500.0, 900.0)));
    }

    #[test]
    fn test_wrap_position_both_overflow() {
        let pos = vec2(1200.0, 1300.0);
        let wrapped = wrap_position(pos, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(wrapped, vec2(200.0, 300.0)));
    }

    #[test]
    fn test_wrap_position_corner_underflow() {
        let pos = vec2(-200.0, -150.0);
        let wrapped = wrap_position(pos, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(wrapped, vec2(800.0, 850.0)));
    }

    // ==================== wrapped_distance_vector tests ====================

    #[test]
    fn test_wrapped_delta_no_wrap_needed() {
        let a = vec2(400.0, 400.0);
        let b = vec2(600.0, 600.0);
        let delta = wrapped_distance_vector(a, b, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(delta, vec2(200.0, 200.0)));
    }

    #[test]
    fn test_wrapped_delta_wrap_x_positive() {
        // a is at x=900, b is at x=100
        // Direct distance: 100 - 900 = -800
        // Wrapped distance: should go right across border = 200
        let a = vec2(900.0, 500.0);
        let b = vec2(100.0, 500.0);
        let delta = wrapped_distance_vector(a, b, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(delta, vec2(200.0, 0.0)));
    }

    #[test]
    fn test_wrapped_delta_wrap_x_negative() {
        // a is at x=100, b is at x=900
        // Direct distance: 900 - 100 = 800
        // Wrapped distance: should go left across border = -200
        let a = vec2(100.0, 500.0);
        let b = vec2(900.0, 500.0);
        let delta = wrapped_distance_vector(a, b, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(delta, vec2(-200.0, 0.0)));
    }

    #[test]
    fn test_wrapped_delta_wrap_y_positive() {
        let a = vec2(500.0, 950.0);
        let b = vec2(500.0, 50.0);
        let delta = wrapped_distance_vector(a, b, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(delta, vec2(0.0, 100.0)));
    }

    #[test]
    fn test_wrapped_delta_wrap_y_negative() {
        let a = vec2(500.0, 50.0);
        let b = vec2(500.0, 950.0);
        let delta = wrapped_distance_vector(a, b, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(delta, vec2(0.0, -100.0)));
    }

    #[test]
    fn test_wrapped_delta_wrap_corner() {
        // a at corner (950, 950), b at corner (50, 50)
        // Should wrap both x and y to get shortest path
        let a = vec2(950.0, 950.0);
        let b = vec2(50.0, 50.0);
        let delta = wrapped_distance_vector(a, b, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(delta, vec2(100.0, 100.0)));
    }

    // ==================== wrapped_distance_abs tests ====================

    #[test]
    fn test_wrapped_distance_same_point() {
        let a = vec2(500.0, 500.0);
        let dist = wrapped_distance_abs(a, a, WORLD_W, WORLD_H);
        assert!(approx_eq(dist, 0.0));
    }

    #[test]
    fn test_wrapped_distance_no_wrap() {
        let a = vec2(0.0, 0.0);
        let b = vec2(100.0, 0.0);
        let dist = wrapped_distance_abs(a, b, WORLD_W, WORLD_H);
        assert!(approx_eq(dist, 100.0));
    }

    #[test]
    fn test_wrapped_distance_across_x_border() {
        // Points near opposite edges should have small wrapped distance
        let a = vec2(10.0, 500.0);
        let b = vec2(990.0, 500.0);
        let dist = wrapped_distance_abs(a, b, WORLD_W, WORLD_H);
        assert!(approx_eq(dist, 20.0));
    }

    #[test]
    fn test_wrapped_distance_across_y_border() {
        let a = vec2(500.0, 5.0);
        let b = vec2(500.0, 995.0);
        let dist = wrapped_distance_abs(a, b, WORLD_W, WORLD_H);
        assert!(approx_eq(dist, 10.0));
    }

    #[test]
    fn test_wrapped_distance_across_corner() {
        // Points at opposite corners, close via wrapping
        let a = vec2(10.0, 10.0);
        let b = vec2(990.0, 990.0);
        let dist = wrapped_distance_abs(a, b, WORLD_W, WORLD_H);
        // dx = 20, dy = 20, distance = sqrt(800) ≈ 28.28
        let expected = (20.0_f32.powi(2) + 20.0_f32.powi(2)).sqrt();
        assert!(approx_eq(dist, expected));
    }

    #[test]
    fn test_wrapped_distance_half_world() {
        // At exactly half the world width, both paths are equal
        let a = vec2(0.0, 500.0);
        let b = vec2(500.0, 500.0);
        let dist = wrapped_distance_abs(a, b, WORLD_W, WORLD_H);
        assert!(approx_eq(dist, 500.0));
    }

    #[test]
    fn test_wrapped_distance_symmetry() {
        let a = vec2(100.0, 200.0);
        let b = vec2(900.0, 800.0);
        let dist_ab = wrapped_distance_abs(a, b, WORLD_W, WORLD_H);
        let dist_ba = wrapped_distance_abs(b, a, WORLD_W, WORLD_H);
        assert!(approx_eq(dist_ab, dist_ba));
    }

    // ==================== normalize_angle tests ====================

    #[test]
    fn test_normalize_angle_zero() {
        assert!(approx_eq(normalize_angle(0.0), 0.0));
    }

    #[test]
    fn test_normalize_angle_positive_within_range() {
        let angle = std::f32::consts::FRAC_PI_2;
        assert!(approx_eq(normalize_angle(angle), angle));
    }

    #[test]
    fn test_normalize_angle_negative_within_range() {
        let angle = -std::f32::consts::FRAC_PI_2;
        assert!(approx_eq(normalize_angle(angle), angle));
    }

    #[test]
    fn test_normalize_angle_over_pi() {
        // 3*PI/2 should wrap to -PI/2
        let angle = 3.0 * std::f32::consts::FRAC_PI_2;
        let expected = -std::f32::consts::FRAC_PI_2;
        assert!(approx_eq(normalize_angle(angle), expected));
    }

    #[test]
    fn test_normalize_angle_under_minus_pi() {
        // -3*PI/2 should wrap to PI/2
        let angle = -3.0 * std::f32::consts::FRAC_PI_2;
        let expected = std::f32::consts::FRAC_PI_2;
        assert!(approx_eq(normalize_angle(angle), expected));
    }

    #[test]
    fn test_normalize_angle_two_pi() {
        // 2*PI should wrap to 0
        let angle = 2.0 * std::f32::consts::PI;
        assert!(approx_eq(normalize_angle(angle), 0.0));
    }
}
