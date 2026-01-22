use crate::brain_neural_network::NeuralNetwork;
use crate::settings::*;
use crate::spatial_hash::HasPos;

use ::rand::Rng;
use macroquad::prelude::*;
use std::iter::IntoIterator;

use std::collections::HashSet;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

const TWO_PI: f32 = 2.0 * PI;

// -------------------- Helpers --------------------

#[inline]
fn next_id() -> usize {
    // Single-threaded game loop: Relaxed is enough.
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[inline]
pub fn wrap_position(pos: Vec2, width: f32, height: f32) -> Vec2 {
    vec2(pos.x.rem_euclid(width), pos.y.rem_euclid(height))
}

/// Compute the shortest delta vector from `a` to `b` in a toroidal world.
/// Returns a vector pointing from `a` towards `b` via the shortest path (possibly across borders).
#[inline]
pub fn wrapped_distance_vector(a: Vec2, b: Vec2, width: f32, height: f32) -> Vec2 {
    let mut dx = b.x - a.x;
    let mut dy = b.y - a.y;

    // Wrap dx to [-width/2, width/2]
    if dx > width / 2.0 {
        dx -= width;
    } else if dx < -width / 2.0 {
        dx += width;
    }

    // Wrap dy to [-height/2, height/2]
    if dy > height / 2.0 {
        dy -= height;
    } else if dy < -height / 2.0 {
        dy += height;
    }

    vec2(dx, dy)
}

/// Compute the shortest distance between two points in a toroidal world.
#[inline]
pub fn wrapped_distance_abs(a: Vec2, b: Vec2, width: f32, height: f32) -> f32 {
    wrapped_distance_vector(a, b, width, height).length()
}

#[inline]
pub fn normalize_angle(angle: f32) -> f32 {
    (angle + PI).rem_euclid(TWO_PI) - PI
}

#[inline]
fn angle_lerp(a: f32, b: f32, t: f32) -> f32 {
    // Simple linear interpolation, then normalized to [-PI, PI]
    normalize_angle(a + (b - a) * t)
}

// -------------------- Shared Core --------------------

#[derive(Clone)]
pub struct AnimalCore {
    pub id: usize,
    pub pos: Vec2,
    pub angle: f32,
    pub energy: f32,
    pub brain: NeuralNetwork,
}

impl AnimalCore {
    pub fn new_with_brain(pos: Vec2, angle: f32, energy: f32, brain: NeuralNetwork) -> Self {
        Self {
            id: next_id(),
            pos,
            angle,
            energy,
            brain,
        }
    }

    #[inline]
    pub fn x(&self) -> f32 {
        self.pos.x
    }

    #[inline]
    pub fn y(&self) -> f32 {
        self.pos.y
    }

    #[inline]
    pub fn set_xy(&mut self, x: f32, y: f32) {
        self.pos = vec2(x, y);
    }
}

#[inline]
fn inherited_brain_with_mutations<R: Rng>(parent: &NeuralNetwork, rng: &mut R) -> NeuralNetwork {
    let mut brain = parent.clone();
    let k = rng.gen_range(2..=6);
    for _ in 0..k {
        brain.mutate(rng);
    }
    brain
}

#[inline]
fn move_with_speed_factor(
    core: &mut AnimalCore,
    speed_factor: f32,
    turn_delta: f32,
    speed: f32,
    moving_decay: f32,
    max_energy: f32,
) {
    // Turning
    core.angle = normalize_angle(core.angle + turn_delta);

    // Moving
    core.pos.x += speed_factor * speed * core.angle.cos();
    core.pos.y += speed_factor * speed * core.angle.sin();

    // Wrap world
    core.pos = wrap_position(core.pos, SCREEN_WIDTH as f32, SCREEN_HEIGHT as f32);

    // Movement energy cost
    core.energy -= (speed_factor * speed_factor) * moving_decay;

    // Keep energy from exploding upward in weird future edits
    if core.energy > max_energy {
        core.energy = max_energy;
    }
}

// -------------------- Predator --------------------

#[derive(Clone)]
pub struct Predator {
    pub core: AnimalCore,
    pub eaten_prey: i32,
    pub repro_cooldown: i32, // frames bis fressen wieder erlaubt
}

impl HasPos for Predator {
    fn x(&self) -> f32 {
        self.core.x()
    }
    fn y(&self) -> f32 {
        self.core.y()
    }
    fn set_pos(&mut self, x: f32, y: f32) {
        self.core.set_xy(x, y);
    }
}

impl Predator {
    pub fn new<R: Rng>(x: f32, y: f32, rng: &mut R) -> Self {
        let angle = rng.gen_range(0.0..TWO_PI);
        let brain = NeuralNetwork::new(NUMBER_SIGHTS_PREDATOR, 2, pred_init_mut(), bias(), rng);

        Self {
            core: AnimalCore::new_with_brain(vec2(x, y), angle, PRED_ENERGY, brain),
            eaten_prey: 0,
            repro_cooldown: 0,
        }
    }

    #[inline]
    pub fn id(&self) -> usize {
        self.core.id
    }

    /// Predator senses prey in a forward cone (±30°) with NUMBER_SIGHTS_PREDATOR rays.
    pub fn get_inputs<'a, I>(&self, preys: I) -> Vec<f32>
    where
        I: IntoIterator<Item = &'a Prey>,
    {
        let n = NUMBER_SIGHTS_PREDATOR.max(1);
        let mut inputs = vec![0.0; n];

        let start_angle = normalize_angle(self.core.angle - 30.0_f32.to_radians());
        let end_angle = normalize_angle(self.core.angle + 30.0_f32.to_radians());

        let predator_pos = self.core.pos;
        let world_w = SCREEN_WIDTH as f32;
        let world_h = SCREEN_HEIGHT as f32;

        for prey in preys {
            let prey_pos = prey.core.pos;
            // Use wrapped distance for toroidal world
            let delta = wrapped_distance_vector(predator_pos, prey_pos, world_w, world_h);
            let dist = delta.length();

            if dist <= 0.0 || dist >= SIGHT_RANGE_PREDATOR {
                continue;
            }

            // Use wrapped delta for angle calculation
            let angle_to_prey = delta.y.atan2(delta.x);

            // Angular width of prey "disc" at distance dist.
            let mut val = PREY_RADIUS / dist;
            if val > 1.0 {
                val = 1.0;
            }
            let angular_width = val.asin();

            for i in 0..n {
                if inputs[i] == 1.0 {
                    continue;
                }
                let t = if n > 1 {
                    i as f32 / (n as f32 - 1.0)
                } else {
                    0.0
                };
                let ray_angle = angle_lerp(start_angle, end_angle, t);
                let diff = normalize_angle(ray_angle - angle_to_prey).abs();
                if diff < angular_width {
                    inputs[i] = 1.0;
                }
            }
        }

        inputs
    }

    pub fn move_step(&mut self, inputs: &[f32]) {
        // Default decay each frame
        self.core.energy -= PRED_DEFAULT_DECAY;

        let energy_ratio = self.core.energy / PRED_ENERGY;
        let outputs = self.core.brain.forward_vectorized(inputs, energy_ratio);

        let speed_factor = outputs[0];
        let turn_delta = outputs[1] * TWO_PI;

        move_with_speed_factor(
            &mut self.core,
            speed_factor,
            turn_delta,
            PREDATOR_SPEED,
            PRED_MOVING_DECAY,
            PRED_ENERGY,
        );
    }

    pub fn hunt_nearby<'a, R: Rng, I>(
        &mut self,
        prey_candidates: I,
        eaten_prey_ids: &mut HashSet<usize>,
        newborn_preds: &mut Vec<Predator>,
        rng: &mut R,
    ) where
        I: IntoIterator<Item = &'a Prey>,
    {
        let eat_r = PREDATOR_RADIUS;
        let predator_pos = self.core.pos;
        let world_w = SCREEN_WIDTH as f32;
        let world_h = SCREEN_HEIGHT as f32;

        for prey in prey_candidates {
            let id = prey.core.id;
            if eaten_prey_ids.contains(&id) {
                continue;
            }

            let prey_pos = prey.core.pos;
            let dist = wrapped_distance_abs(predator_pos, prey_pos, world_w, world_h);

            if dist < eat_r {
                eaten_prey_ids.insert(id);

                self.core.energy = (self.core.energy + PREDATOR_ENERGY_GAIN).min(PRED_ENERGY);

                self.eaten_prey += 1;

                if let Some(child) = self.reproduce(rng) {
                    newborn_preds.push(child);
                }

                // If you want "max one kill per predator per frame", uncomment:
                // break;
            }
        }
    }

    pub fn reproduce<R: Rng>(&mut self, rng: &mut R) -> Option<Predator> {
        const REPRO_COOLDOWN_FRAMES: i32 = 5;

        if self.repro_cooldown > 0 {
            return None;
        }

        if self.eaten_prey < 3 {
            return None;
        }

        self.eaten_prey = 0;
        self.repro_cooldown = REPRO_COOLDOWN_FRAMES;

        // Tiny offset near parent
        let ox = self.core.pos.x + rng.gen_range(-1..=1) as f32;
        let oy = self.core.pos.y + rng.gen_range(-1..=1) as f32;

        // Child gets parent's brain + mutations
        let mut child = Predator::new(ox, oy, rng);
        child.core.brain = inherited_brain_with_mutations(&self.core.brain, rng);

        // You can decide if newborn starts full or some split. Keeping your previous behavior:
        child.core.energy = PRED_ENERGY;

        Some(child)
    }

}

// -------------------- Prey --------------------

#[derive(Clone)]
pub struct Prey {
    pub core: AnimalCore,
    pub rest_time: i32,
}

impl HasPos for Prey {
    fn x(&self) -> f32 {
        self.core.x()
    }
    fn y(&self) -> f32 {
        self.core.y()
    }
    fn set_pos(&mut self, x: f32, y: f32) {
        self.core.set_xy(x, y);
    }
}

impl Prey {
    pub fn new<R: Rng>(x: f32, y: f32, rng: &mut R) -> Self {
        let angle = rng.gen_range(0.0..TWO_PI);
        let brain = NeuralNetwork::new(NUMBER_SIGHTS_PREY, 2, prey_init_mut(), bias(), rng);

        Self {
            core: AnimalCore::new_with_brain(vec2(x, y), angle, PREY_ENERGY, brain),
            rest_time: 0,
        }
    }

    #[inline]
    pub fn id(&self) -> usize {
        self.core.id
    }

    /// Prey senses predators in 360° sectors (NUMBER_SIGHTS_PREY bins).
    pub fn sense_predators<'a, I>(&self, predators: I) -> Vec<f32>
    where
        I: IntoIterator<Item = &'a Predator>,
    {
        let n = NUMBER_SIGHTS_PREY.max(1);
        let mut inputs = vec![0.0; n];

        let sector_size = TWO_PI / (n as f32);
        let prey_pos = self.core.pos;
        let world_w = SCREEN_WIDTH as f32;
        let world_h = SCREEN_HEIGHT as f32;

        for pred in predators {
            let pred_pos = pred.core.pos;
            // Use wrapped distance for toroidal world
            let delta = wrapped_distance_vector(prey_pos, pred_pos, world_w, world_h);
            let dist = delta.length();

            if dist >= SIGHT_RANGE_PREY {
                continue;
            }

            // Use wrapped delta for angle calculation
            let angle_to_pred = delta.y.atan2(delta.x);
            let rel = normalize_angle(angle_to_pred - self.core.angle); // [-PI, PI]

            // Map [-PI, PI] -> [0, TWO_PI)
            let shifted = rel + PI;
            let idx = (shifted / sector_size).floor() as i32;
            let idx = idx.rem_euclid(n as i32) as usize;

            inputs[idx] = 1.0;
        }

        inputs
    }

    pub fn move_step(&mut self, inputs: &[f32]) {
        let energy_ratio = self.core.energy / PREY_ENERGY;
        let outputs = self.core.brain.forward_vectorized(inputs, energy_ratio);

        let speed_factor = outputs[0];

        // Rest threshold: if barely moving, recover energy and do not move.
        if speed_factor < 0.05 {
            self.core.energy = (self.core.energy + PREY_REST_ENERGY_GAIN).min(PREY_ENERGY);
            return;
        }

        if self.core.energy < 0.0 {
            return;
        }

        let turn_delta = outputs[1] * PI;

        move_with_speed_factor(
            &mut self.core,
            speed_factor,
            turn_delta,
            PREY_SPEED,
            PREY_MOVING_DECAY,
            PREY_ENERGY,
        );
    }

    pub fn reproduce<R: Rng>(&mut self, rng: &mut R, has_slot: bool) -> Option<Prey> {
        self.rest_time += 1;

        let threshold = (PREY_REPRODUCATION_RATE * FRAMES_PER_SECOND as f32) as i32;

        if self.rest_time < threshold {
            return None;
        }

        // Jetzt wäre sie "bereit". Wenn aber kein Slot frei ist:
        if !has_slot {
            // nicht zurücksetzen, sonst verliert sie den "ready"-Status
            self.rest_time = threshold; // clamp
            return None;
        }

        // Slot ist frei -> Geburt
        self.rest_time = 0;

        let ox =
            rng.gen_range((self.core.pos.x as i32 - 50)..=(self.core.pos.x as i32 + 50)) as f32;
        let oy =
            rng.gen_range((self.core.pos.y as i32 - 50)..=(self.core.pos.y as i32 + 50)) as f32;

        let mut child = Prey::new(ox, oy, rng);
        child.core.brain = inherited_brain_with_mutations(&self.core.brain, rng);

        Some(child)
    }

}

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

    // ==================== wrap_position tests ====================

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
