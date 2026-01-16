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

#[inline]
pub fn distance(a: Vec2, b: Vec2) -> f32 {
    (a - b).length()
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

        for prey in preys {
            let prey_pos = prey.core.pos;
            let dist = distance(predator_pos, prey_pos);

            if dist <= 0.0 || dist >= SIGHT_RANGE_PREDATOR {
                continue;
            }

            let angle_to_prey = (prey_pos.y - predator_pos.y).atan2(prey_pos.x - predator_pos.x);

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
        let eat_r = PREDATOR_RADIUS + PREY_RADIUS;
        let predator_pos = self.core.pos;

        for prey in prey_candidates {
            let id = prey.core.id;
            if eaten_prey_ids.contains(&id) {
                continue;
            }

            let prey_pos = prey.core.pos;
            let dist = distance(predator_pos, prey_pos);

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
        if self.eaten_prey <= 3 {
            return None;
        }

        self.eaten_prey = 0;

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

    pub fn draw_sight(&self) {
        let n = NUMBER_SIGHTS_PREDATOR.max(1);

        let start_angle = self.core.angle - 30.0_f32.to_radians();
        let end_angle = self.core.angle + 30.0_f32.to_radians();

        for i in 0..n {
            let t = if n > 1 {
                i as f32 / (n as f32 - 1.0)
            } else {
                0.0
            };
            let sight_angle = start_angle + t * (end_angle - start_angle);

            let end_x = self.core.pos.x + SIGHT_RANGE_PREDATOR * sight_angle.cos();
            let end_y = self.core.pos.y + SIGHT_RANGE_PREDATOR * sight_angle.sin();

            draw_line(self.core.pos.x, self.core.pos.y, end_x, end_y, 1.0, YELLOW);
        }
    }

    pub fn draw(&self) {
        draw_circle(self.core.pos.x, self.core.pos.y, PREDATOR_RADIUS, RED);
        self.draw_sight();
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

        for pred in predators {
            let pred_pos = pred.core.pos;
            let dist = distance(prey_pos, pred_pos);

            if dist >= SIGHT_RANGE_PREY {
                continue;
            }

            let angle_to_pred = (pred_pos.y - prey_pos.y).atan2(pred_pos.x - prey_pos.x);
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

    pub fn reproduce<R: Rng>(&mut self, rng: &mut R) -> Option<Prey> {
        self.rest_time += 1;

        let threshold = (PREY_REPRODUCATION_RATE * FRAMES_PER_SECOND as f32) as i32;

        if self.rest_time < threshold {
            return None;
        }

        self.rest_time = 0;

        // Random offset within +-50 px
        let ox =
            rng.gen_range((self.core.pos.x as i32 - 50)..=(self.core.pos.x as i32 + 50)) as f32;
        let oy =
            rng.gen_range((self.core.pos.y as i32 - 50)..=(self.core.pos.y as i32 + 50)) as f32;

        let mut child = Prey::new(ox, oy, rng);
        child.core.brain = inherited_brain_with_mutations(&self.core.brain, rng);

        Some(child)
    }

    pub fn draw_sight(&self) {
        let n = NUMBER_SIGHTS_PREY.max(1);
        let step = TWO_PI / (n as f32);

        for i in 0..n {
            let sight_angle = self.core.angle + step * (i as f32);

            let end_x = self.core.pos.x + SIGHT_RANGE_PREY * sight_angle.cos();
            let end_y = self.core.pos.y + SIGHT_RANGE_PREY * sight_angle.sin();

            draw_line(self.core.pos.x, self.core.pos.y, end_x, end_y, 1.0, SKYBLUE);
        }
    }

    pub fn draw(&self) {
        draw_circle(self.core.pos.x, self.core.pos.y, PREY_RADIUS, GREEN);
        self.draw_sight();
    }
}
