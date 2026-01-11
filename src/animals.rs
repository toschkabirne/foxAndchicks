use crate::brain_neural_network::{round_ties_even_to_i32, NeuralNetwork};
use crate::settings;
use crate::spatial_hash::HasPos;
use ::rand::Rng;
use macroquad::prelude::*;

use std::cell::RefCell;
use std::f32::consts::PI;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

// ---- Helpers (Python equivalents) ----
pub fn wrap_position(pos: (f32, f32), width: f32, height: f32) -> (f32, f32) {
    let (x, y) = pos;
    (x.rem_euclid(width), y.rem_euclid(height))
}

pub fn distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

pub fn normalize_angle(angle: f32) -> f32 {
    (angle + PI).rem_euclid(2.0 * PI) - PI
}

pub const PREDATOR_RADIUS: f32 = 10.0;
pub const PREY_RADIUS: f32 = 7.0;

// -------------------- Predator --------------------
#[derive(Clone)]
pub struct Predator {
    pub id: usize,
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub energy: f32,
    pub brain: Rc<RefCell<NeuralNetwork>>,
    pub eaten_prey: i32,
}

impl HasPos for Predator {
    fn x(&self) -> f32 {
        self.x
    }
    fn y(&self) -> f32 {
        self.y
    }
    fn set_pos(&mut self, x: f32, y: f32) {
        self.x = x;
        self.y = y;
    }
}

impl Predator {
    pub fn new<R: Rng>(x: f32, y: f32, rng: &mut R) -> Self {
        let angle = rng.gen_range(0.0..(2.0 * PI));
        let brain = Rc::new(RefCell::new(NeuralNetwork::new(
            settings::NUMBER_SIGHTS_PREDATOR,
            2,
            settings::pred_init_mut(),
            settings::bias(),
            rng,
        )));

        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);

        Self {
            id,
            x,
            y,
            angle,
            energy: settings::PRED_ENERGY,
            brain,
            eaten_prey: 0,
        }
    }

    pub fn get_inputs(&self, preys: &[Rc<RefCell<Prey>>]) -> Vec<f32> {
        let mut inputs = vec![0.0; settings::NUMBER_SIGHTS_PREDATOR];
        let start_angle = self.angle - 30.0_f32.to_radians();

        let ray_angles: Vec<f32> = (0..settings::NUMBER_SIGHTS_PREDATOR)
            .map(|i| normalize_angle(start_angle + (i as f32) * 6.0_f32.to_radians()))
            .collect();

        for prey_rc in preys {
            let prey = prey_rc.borrow();
            let dist = distance((self.x, self.y), (prey.x, prey.y));
            if dist < settings::SIGHT_RANGE_PREDATOR && dist > 0.0 {
                let angle_to_prey = (prey.y - self.y).atan2(prey.x - self.x);

                let mut val = PREY_RADIUS / dist;
                if val > 1.0 {
                    val = 1.0;
                }
                let angular_width = val.asin();

                for i in 0..settings::NUMBER_SIGHTS_PREDATOR {
                    if inputs[i] == 1.0 {
                        continue;
                    }
                    let diff = normalize_angle(ray_angles[i] - angle_to_prey).abs();
                    if diff < angular_width {
                        inputs[i] = 1.0;
                    }
                }
            }
        }

        inputs
    }

    pub fn move_step(&mut self, inputs: &[f32]) {
        self.energy -= settings::PRED_DEFAULT_DECAY;

        let outputs = self
            .brain
            .borrow_mut()
            .forward_vectorized(inputs, self.energy / settings::PRED_ENERGY);
        let speed_factor = outputs[0];

        self.angle += outputs[1] * PI;

        self.x += speed_factor * settings::PREDATOR_SPEED * self.angle.cos();
        self.y += speed_factor * settings::PREDATOR_SPEED * self.angle.sin();

        let (nx, ny) = wrap_position(
            (self.x, self.y),
            settings::SCREEN_WIDTH as f32,
            settings::SCREEN_HEIGHT as f32,
        );
        self.x = nx;
        self.y = ny;

        self.energy -= (speed_factor * speed_factor) * settings::PRED_MOVING_DECAY;
    }

    pub fn hunt(&mut self, preys: &mut Vec<Rc<RefCell<Prey>>>) {
        let mut i = 0;
        while i < preys.len() {
            let is_eaten = {
                let prey = preys[i].borrow();
                distance((self.x, self.y), (prey.x, prey.y)) < 10.0
            };

            if is_eaten {
                self.energy = self
                    .energy
                    .min(settings::PRED_ENERGY)
                    .max(self.energy + settings::PREDATOR_ENERGY_GAIN);
                self.eaten_prey += 1;
                preys.remove(i);
            } else {
                i += 1;
            }
        }
    }

    pub fn reproduce<R: Rng>(&mut self, rng: &mut R) -> Option<Rc<RefCell<Predator>>> {
        if self.eaten_prey > 3 {
            self.eaten_prey = 0;

            let ox = self.x + rng.gen_range(-1..=1) as f32;
            let oy = self.y + rng.gen_range(-1..=1) as f32;

            let mut offspring = Predator::new(ox, oy, rng);

            // Python: offspring.brain = self.brain (shared reference)
            offspring.brain = Rc::clone(&self.brain);

            for _ in 0..rng.gen_range(2..=6) {
                offspring.brain.borrow_mut().mutate(rng);
            }

            return Some(Rc::new(RefCell::new(offspring)));
        }
        None
    }

    pub fn draw_sight(&self) {
        let start_angle = self.angle - 30.0_f32.to_radians();
        let end_angle = self.angle + 30.0_f32.to_radians();

        for i in 0..settings::NUMBER_SIGHTS_PREDATOR {
            let t = if settings::NUMBER_SIGHTS_PREDATOR > 1 {
                i as f32 / (settings::NUMBER_SIGHTS_PREDATOR as f32 - 1.0)
            } else {
                0.0
            };
            let sight_angle = start_angle + t * (end_angle - start_angle);

            let end_x = self.x + settings::SIGHT_RANGE_PREDATOR * sight_angle.cos();
            let end_y = self.y + settings::SIGHT_RANGE_PREDATOR * sight_angle.sin();

            draw_line(self.x, self.y, end_x, end_y, 1.0, YELLOW);
        }
    }

    pub fn draw(&self) {
        draw_circle(self.x, self.y, PREDATOR_RADIUS, RED);
        self.draw_sight();
    }
}

// -------------------- Prey --------------------
#[derive(Clone)]
pub struct Prey {
    pub id: usize,
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub energy: f32,
    pub rest_time: i32,
    pub brain: Rc<RefCell<NeuralNetwork>>,
}

impl HasPos for Prey {
    fn x(&self) -> f32 {
        self.x
    }
    fn y(&self) -> f32 {
        self.y
    }
    fn set_pos(&mut self, x: f32, y: f32) {
        self.x = x;
        self.y = y;
    }
}

impl Prey {
    pub fn new<R: Rng>(x: f32, y: f32, rng: &mut R) -> Self {
        let angle = rng.gen_range(0.0..(2.0 * PI));
        let brain = Rc::new(RefCell::new(NeuralNetwork::new(
            settings::NUMBER_SIGHTS_PREY,
            2,
            settings::prey_init_mut(),
            settings::bias(),
            rng,
        )));

        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);

        Self {
            id,
            x,
            y,
            angle,
            energy: settings::PREY_ENERGY,
            rest_time: 0,
            brain,
        }
    }

    // Python: get_inputs(self, spatialpredators, predators)
    // Returns None if eaten.
    pub fn get_inputs<R: Rng>(
        &self,
        spatialpredators: &[Rc<RefCell<Predator>>],
        predators_vec: &mut Vec<Rc<RefCell<Predator>>>,
        rng: &mut R,
    ) -> Option<Vec<f32>> {
        let mut inputs = vec![0.0; settings::NUMBER_SIGHTS_PREY];
        let sector_size = (360.0 / settings::NUMBER_SIGHTS_PREY as f32).to_radians();

        for pred_rc in spatialpredators {
            let mut predator = pred_rc.borrow_mut();

            let dx = predator.x - self.x;
            let dy = predator.y - self.y;
            let dist_sq = dx * dx + dy * dy;

            // Eating logic: if dist_sq < 100 (10^2)
            if dist_sq < 100.0 {
                predator.energy = predator
                    .energy
                    .min(settings::PRED_ENERGY)
                    .max(predator.energy + settings::PREDATOR_ENERGY_GAIN);
                predator.eaten_prey += 1;

                if let Some(new_pred) = predator.reproduce(rng) {
                    predators_vec.push(new_pred);
                }

                return None; // eaten
            }

            if dist_sq < settings::SIGHT_RANGE_PREY * settings::SIGHT_RANGE_PREY {
                let angle_to_pred = dy.atan2(dx);

                let mut rel_angle = angle_to_pred - self.angle;
                rel_angle = (rel_angle + PI).rem_euclid(2.0 * PI) - PI;

                // idx = int(round(rel_angle / sector_size)) % NUMBER_SIGHTS_PREY
                let raw = rel_angle / sector_size;
                let idx = round_ties_even_to_i32(raw)
                    .rem_euclid(settings::NUMBER_SIGHTS_PREY as i32)
                    as usize;

                inputs[idx] = 1.0;
            }
        }

        Some(inputs)
    }

    pub fn move_step(&mut self, inputs: &[f32]) {
        let outputs = self
            .brain
            .borrow_mut()
            .forward_vectorized(inputs, self.energy / settings::PREY_ENERGY);
        let speed_factor = outputs[0];

        // rest threshold
        if speed_factor < 0.05 {
            self.energy =
                (self.energy + settings::PREY_REST_ENERGY_GAIN).min(settings::PREY_ENERGY);
            return;
        }

        if self.energy < 0.0 {
            return;
        }

        self.angle += outputs[1] * PI;

        self.x += speed_factor * settings::PREY_SPEED * self.angle.cos();
        self.y += speed_factor * settings::PREY_SPEED * self.angle.sin();

        let (nx, ny) = wrap_position(
            (self.x, self.y),
            settings::SCREEN_WIDTH as f32,
            settings::SCREEN_HEIGHT as f32,
        );
        self.x = nx;
        self.y = ny;

        self.energy -= (speed_factor * speed_factor) * settings::PREY_MOVING_DECAY;
    }

    pub fn reproduce<R: Rng>(&mut self, rng: &mut R) -> Option<Rc<RefCell<Prey>>> {
        self.rest_time += 1;
        let threshold =
            (settings::PREY_REPRODUCATION_RATE * settings::FRAMES_PER_SECOND as f32) as i32;

        if self.rest_time >= threshold {
            self.rest_time = 0;

            let ox = rng.gen_range((self.x as i32 - 50)..=(self.x as i32 + 50)) as f32;
            let oy = rng.gen_range((self.y as i32 - 50)..=(self.y as i32 + 50)) as f32;

            let mut offspring = Prey::new(ox, oy, rng);

            // Python: offspring.brain = self.brain (shared reference)
            offspring.brain = Rc::clone(&self.brain);

            for _ in 0..rng.gen_range(2..=6) {
                offspring.brain.borrow_mut().mutate(rng);
            }

            return Some(Rc::new(RefCell::new(offspring)));
        }

        None
    }

    pub fn draw_sight(&self) {
        for i in 0..settings::NUMBER_SIGHTS_PREY {
            let sight_angle =
                self.angle + (360.0 / settings::NUMBER_SIGHTS_PREY as f32).to_radians() * i as f32;

            let end_x = self.x + settings::SIGHT_RANGE_PREY * sight_angle.cos();
            let end_y = self.y + settings::SIGHT_RANGE_PREY * sight_angle.sin();

            draw_line(self.x, self.y, end_x, end_y, 1.0, SKYBLUE);
        }
    }

    pub fn draw(&self) {
        draw_circle(self.x, self.y, PREY_RADIUS, GREEN);
        self.draw_sight();
    }
}
