use crate::animals::{Predator, Prey};
use crate::data_manager::{DataManager, Frame};
use crate::settings::{self};
use crate::spatial_hash::SpatialHash;
use crate::visualization::{draw_frame, draw_game_stats};
use ::rand::rngs::ThreadRng;
use ::rand::Rng;
use macroquad::prelude::*;
use std::collections::HashSet;

// main structs and logic for the game itself -> main should be light, just setup and loop
pub struct Game {
    pub frame_count: usize,
    predators: Vec<Predator>,
    preys: Vec<Prey>,
    spatial_hash_preds: SpatialHash,
    spatial_hash_preys: SpatialHash,
    max_predators: usize,
    max_preys: usize,
    data_manager: DataManager,
    scratch_idxs: Vec<usize>,
}

impl Game {
    pub fn predator_count(&self) -> usize {
        self.predators.len()
    }
    pub fn prey_count(&self) -> usize {
        self.preys.len()
    }
    /// Returns the actual filename (with timestamp) used for storing data
    pub fn get_data_filename(&self) -> &str {
        &self.data_manager.filename
    }
}

impl Game {
    /// Creates a new Game with custom parameters
    pub fn new(
        file_name: &str,
        num_preds: usize,
        num_preys: usize,
        max_preds: usize,
        max_preys: usize,
    ) -> Self {
        let mut rng = ::rand::thread_rng();

        // Spawn initial predators and preys as Rc<RefCell<>> for shared mutability
        let predators: Vec<Predator> = (0..num_preds)
            .map(|_| {
                Predator::new(
                    rng.gen_range(0.0..settings::SCREEN_WIDTH as f32),
                    rng.gen_range(0.0..settings::SCREEN_HEIGHT as f32),
                    &mut rng,
                )
            })
            .collect();

        let preys: Vec<Prey> = (0..num_preys)
            .map(|_| {
                Prey::new(
                    rng.gen_range(0.0..settings::SCREEN_WIDTH as f32),
                    rng.gen_range(0.0..settings::SCREEN_HEIGHT as f32),
                    &mut rng,
                )
            })
            .collect();

        // Set up spatial hash
        let cell_pred =
            ((settings::SCREEN_WIDTH as f32) / settings::SIGHT_RANGE_PREDATOR).floor() as i32;
        let cell_prey =
            ((settings::SCREEN_WIDTH as f32) / settings::SIGHT_RANGE_PREY).floor() as i32;

        let spatial_hash_preds: SpatialHash = SpatialHash::new(cell_pred);
        let spatial_hash_preys: SpatialHash = SpatialHash::new(cell_prey);

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
            scratch_idxs: Vec::with_capacity(128),
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
        self.frame_count += 1;
        let mut rng: ThreadRng = ::rand::thread_rng();

        // ----------------------------
        // PREDATOR PHASE
        // ----------------------------

        // Build PREY hash (preys are still at "start of frame" positions)
        self.spatial_hash_preys.rebuild_from(&self.preys);

        let mut eaten_prey_ids: HashSet<usize> = HashSet::new();
        let mut newborn_preds: Vec<Predator> = Vec::new();

        // Drain predators to avoid borrow conflicts (same pattern as before, but now it's cheap)
        let mut surviving_preds: Vec<Predator> = Vec::with_capacity(self.predators.len());

        for mut pred in self.predators.drain(..) {
            if pred.repro_cooldown > 0 {
                pred.repro_cooldown -= 1;
            }
            // Sense nearby preys at current predator position
            let px = pred.core.pos.x;
            let py = pred.core.pos.y;

            self.spatial_hash_preys
                .query_into(&mut self.scratch_idxs, px, py);
            let inputs = pred.get_inputs(self.scratch_idxs.iter().map(|&i| &self.preys[i]));

            pred.move_step(&inputs);

            // Hunt near new position (preys haven't moved yet)
            self.spatial_hash_preys.query_into(
                &mut self.scratch_idxs,
                pred.core.pos.x,
                pred.core.pos.y,
            );
            pred.hunt_nearby(
                self.scratch_idxs.iter().map(|&i| &self.preys[i]),
                &mut eaten_prey_ids,
                &mut newborn_preds,
                &mut rng,
            );

            let dead = pred.core.energy < 0.0;
            if !dead {
                surviving_preds.push(pred);
            }
        }

        // Newborn predators join next frame, but cap to max_predators
        let free_slots = self.max_predators.saturating_sub(surviving_preds.len());
        if free_slots > 0 {
            surviving_preds.extend(newborn_preds.into_iter().take(free_slots));
        }
        self.predators = surviving_preds;

        // Remove eaten preys BEFORE prey phase
        if !eaten_prey_ids.is_empty() {
            self.preys
                .retain(|prey| !eaten_prey_ids.contains(&prey.core.id));
        }

        // ----------------------------
        // PREY PHASE
        // ----------------------------

        // Build PREDATOR hash (AFTER predator movement/removal)
        self.spatial_hash_preds.rebuild_from(&self.predators);

        let base = self.preys.len();
        let allowed_newborns = self.max_preys.saturating_sub(base);

        let mut newborn_preys: Vec<Prey> = Vec::new();

        // Iterate mutably: preys move + may reproduce
        for prey in self.preys.iter_mut() {
            let x = prey.core.pos.x;
            let y = prey.core.pos.y;

            self.spatial_hash_preds
                .query_into(&mut self.scratch_idxs, x, y);
            let inputs =
                prey.sense_predators(self.scratch_idxs.iter().map(|&i| &self.predators[i]));

            prey.move_step(&inputs);

            let has_slot = newborn_preys.len() < allowed_newborns;
            if let Some(child) = prey.reproduce(&mut rng, has_slot) {
                newborn_preys.push(child);
            }
        }

        self.preys.extend(newborn_preys);

        Frame::new(&self.predators, &self.preys, self.frame_count)
    }

    pub fn calculate_and_store_next_frame(&mut self) {
        let frame = self.next_frame();
        self.data_manager.store_frame(&frame);
    }

    pub async fn playback(file_name: &str, draw_sight_lines: bool) {
        for frame in DataManager::read_frames(file_name) {
            clear_background(settings::BACKGROUND_COLOR);
            draw_frame(&frame, draw_sight_lines);

            let (pred_count, prey_count) = frame.counts();
            draw_game_stats(pred_count, prey_count, frame.tick);

            next_frame().await;
        }
    }
}
