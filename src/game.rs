use crate::settings::{self, MAX_PREY_COUNT};
use crate::spatial_hash::SpatialHash;
use crate::animals::{Predator, Prey, PREDATOR_RADIUS, PREY_RADIUS};
use crate::data_manager::{DataManager, AnimalType, Frame};
use macroquad::prelude::*;
use ::rand::Rng;
use ::rand::rngs::ThreadRng;
use std::cell::RefCell;
use std::rc::Rc;

// main structs and logic for the game itself -> main should be light, just setup and loop
pub struct Game{
    pub frame_count: usize,
    predators: Vec<Rc<RefCell<Predator>>>,
    preys: Vec<Rc<RefCell<Prey>>>,
    spatial_hash_preds: SpatialHash<Predator>,
    spatial_hash_preys: SpatialHash<Prey>,
    max_predators: usize,
    max_preys: usize,
    data_manager: DataManager,
}


impl Game {
    /// Creates a new Game with custom parameters
    pub fn new(file_name: &str, num_preds: usize, num_preys: usize, max_preds: usize, max_preys: usize) -> Self {

        let mut rng = ::rand::thread_rng();

        // Spawn initial predators and preys as Rc<RefCell<>> for shared mutability
        let predators: Vec<Rc<RefCell<Predator>>> = (0..num_preds)
            .map(|_| {
                Rc::new(RefCell::new(Predator::new(
                    rng.gen_range(0.0..settings::SCREEN_WIDTH as f32),
                    rng.gen_range(0.0..settings::SCREEN_HEIGHT as f32),
                    &mut rng,
                )))
            })
            .collect();

        let preys: Vec<Rc<RefCell<Prey>>> = (0..num_preys)
            .map(|_| {
                Rc::new(RefCell::new(Prey::new(
                    rng.gen_range(0.0..settings::SCREEN_WIDTH as f32),
                    rng.gen_range(0.0..settings::SCREEN_HEIGHT as f32),
                    &mut rng,
                )))
            })
            .collect();

        // Set up spatial hash
        let cell_pred =
            ((settings::SCREEN_WIDTH as f32) / settings::SIGHT_RANGE_PREDATOR).floor() as i32;
        let cell_prey = ((settings::SCREEN_WIDTH as f32) / settings::SIGHT_RANGE_PREY).floor() as i32;

        let spatial_hash_preds: SpatialHash<Predator> = SpatialHash::new(cell_pred);
        let spatial_hash_preys: SpatialHash<Prey> = SpatialHash::new(cell_prey);

        let data_manager: DataManager = DataManager::new(file_name);

        Game {
            frame_count: 0,
            predators,
            preys,
            spatial_hash_preds,
            spatial_hash_preys,
            max_predators: max_preds,
            max_preys: max_preys,
            data_manager,
        }
    }

    /// Creates a new Game with default settings
    pub fn new_default(file_name: &str) -> Self {
        Game::new(
            file_name,
            settings::PRED_INIT_NUMB,
            settings::PREY_INIT_NUMB,
            settings::MAX_PRED_COUNT,
            settings::MAX_PREY_COUNT,
        )
    }

    pub fn next_frame(&mut self) -> Frame {
        // Logic for updating the game state for the next frame
        self.frame_count += 1;
        let mut rng: ThreadRng = ::rand::thread_rng();

        // Clear spatial hashes
        self.spatial_hash_preds.clear();
        self.spatial_hash_preys.clear();

        // Insert all predators + preys into spatial hashes
        for p in &self.predators {
            self.spatial_hash_preds.insert(Rc::clone(p));
        }
        for pr in &self.preys {
            self.spatial_hash_preys.insert(Rc::clone(pr));
        }

        // --- Predator update/draw ---
        let mut i = 0;
        while i < self.predators.len() {
            let pred_rc = Rc::clone(&self.predators[i]);
            let (px, py) = {
                let pred = pred_rc.borrow();
                (pred.x, pred.y)
            };

            let nearby_preys = self.spatial_hash_preys.query(px, py);

            // get NN inputs from nearby preys and move
            let dead = {
                let mut pred = pred_rc.borrow_mut();
                let inputs = pred.get_inputs(&nearby_preys);
                pred.move_step(&inputs);
                pred.energy < 0.0
            };

            if dead {
                self.predators.remove(i);
            } else {
                i += 1;
            }
        }

        // --- Prey update/draw ---
        let mut j = 0;
        while j < self.preys.len() {
            let prey_rc = Rc::clone(&self.preys[j]);
            let (x, y) = {
                let prey = prey_rc.borrow();
                (prey.x, prey.y)
            };

            let nearby_preds = self.spatial_hash_preds.query(x, y);

            // Get inputs for nearby predators
            let inputs_opt = {
                let prey = prey_rc.borrow();
                prey.get_inputs(&nearby_preds, &mut self.predators, &mut rng)
            };

            let Some(inputs) = inputs_opt else {
                self.preys.remove(j);
                continue;
            };

            {
                let mut prey = prey_rc.borrow_mut();
                prey.move_step(&inputs);

                if self.preys.len() < settings::MAX_PREY_COUNT {
                    if let Some(new_prey) = prey.reproduce(&mut rng) {
                        self.preys.push(new_prey);
                    }
                }
            }
            
            j += 1;
        }

        return Frame::new(&self.predators, &self.preys, self.frame_count);
    }

    pub fn calculate_and_store_next_frame(&mut self) {
        let frame = self.next_frame();
        self.data_manager.store_frame(&frame);
    }

    pub async fn playback(file_name: &str) {
        for frame in DataManager::read_frames(file_name){
            clear_background(settings::BACKGROUND_COLOR);
            frame.draw(true);
            next_frame().await;
        }
    }
}