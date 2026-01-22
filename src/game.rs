use crate::animals::{Predator, Prey};
use crate::data_manager::{DataManager, Frame, IndexedFrameReader};
use crate::settings::{self};
use crate::spatial_hash::SpatialHash;
use crate::visualization::{draw_frame, draw_game_stats, draw_playback_controls, PlaybackState};
use ::rand::Rng;
use macroquad::prelude::*;
use rayon::prelude::*;
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
    data_manager: Option<DataManager>,
    scratch_idxs: Vec<usize>,
}

impl Game {
    pub fn predator_count(&self) -> usize {
        self.predators.len()
    }
    pub fn prey_count(&self) -> usize {
        self.preys.len()
    }
    /// Returns the actual filename (with timestamp) used for storing data, if any
    pub fn get_data_filename(&self) -> Option<&str> {
        self.data_manager.as_ref().map(|dm| dm.filename.as_str())
    }
}

impl Game {
    /// Creates a new Game with custom parameters
    pub fn new(
        file_name: Option<&str>,
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
            ((settings::SCREEN_WIDTH as f32) / settings::PREDATOR_SIGHT_RANGE).floor() as i32;
        let cell_prey =
            ((settings::SCREEN_WIDTH as f32) / settings::PREY_SIGHT_RANGE).floor() as i32;

        let world_w = settings::SCREEN_WIDTH as f32;
        let world_h = settings::SCREEN_HEIGHT as f32;

        let spatial_hash_preds: SpatialHash = SpatialHash::new(cell_pred, world_w, world_h);
        let spatial_hash_preys: SpatialHash = SpatialHash::new(cell_prey, world_w, world_h);

        let data_manager = file_name.map(|name| DataManager::new(name));

        Game {
            frame_count: 0,
            predators,
            preys,
            spatial_hash_preds,
            spatial_hash_preys,
            max_predators: max_preds,
            max_preys: max_preys,
            data_manager,
            scratch_idxs: Vec::new(),
        }
    }

    /// Creates a new Game with default settings
    pub fn new_default(file_name: Option<&str>) -> Self {
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

        // ----------------------------
        // PREDATOR PHASE
        // ----------------------------

        // Build PREY hash (preys are still at "start of frame" positions)
        self.spatial_hash_preys.rebuild_from(&self.preys);

        // Pull shared borrows out of `self` for cleaner parallel closures
        let preys_ref: &Vec<Prey> = &self.preys;
        let hash_preys: &SpatialHash = &self.spatial_hash_preys;

        // PARALLEL: Sense and move predators
        // Each predator independently senses nearby preys and computes its movement
        self.predators.par_iter_mut().for_each_init(
            || Vec::<usize>::new(),
            |scratch, pred| {
                if pred.repro_cooldown > 0 {
                    pred.repro_cooldown -= 1;
                }

                // Sense nearby preys at current predator position
                hash_preys.query_into(scratch, pred.core.pos.x, pred.core.pos.y);

                // SAFETY: We only read from self.preys here, no mutation
                let inputs = pred.get_inputs(scratch.iter().filter_map(|&i| preys_ref.get(i)));

                pred.move_step(inputs);
            },
        );

        // SEQUENTIAL: Hunting phase (requires mutable shared state for eaten_prey_ids)
        let mut eaten_prey_ids: HashSet<usize> = HashSet::new();
        let mut newborn_preds: Vec<Predator> = Vec::new();
        let mut rng = ::rand::thread_rng();

        for pred in self.predators.iter_mut() {
            // Hunt near new position (preys haven't moved yet)
            self.spatial_hash_preys.query_into(
                &mut self.scratch_idxs,
                pred.core.pos.x,
                pred.core.pos.y,
            );

            pred.hunt_nearby(
                self.scratch_idxs.iter().filter_map(|&i| self.preys.get(i)),
                &mut eaten_prey_ids,
                &mut newborn_preds,
                &mut rng,
            );
        }

        // Remove dead predators and add newborns
        self.predators.retain(|pred| pred.core.energy >= 0.0);

        let free_slots = self.max_predators.saturating_sub(self.predators.len());
        if free_slots > 0 {
            self.predators
                .extend(newborn_preds.into_iter().take(free_slots));
        }

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

        // Pull shared borrows out for parallel closures
        let preds_ref: &Vec<Predator> = &self.predators;
        let hash_preds: &SpatialHash = &self.spatial_hash_preds;

        // PARALLEL: Sense and move preys
        self.preys.par_iter_mut().for_each_init(
            || Vec::<usize>::new(),
            |scratch, prey| {
                hash_preds.query_into(scratch, prey.core.pos.x, prey.core.pos.y);

                // SAFETY: We only read from self.predators here, no mutation
                let inputs = prey.get_inputs(scratch.iter().filter_map(|&i| preds_ref.get(i)));

                prey.move_step(inputs);
            },
        );

        // SEQUENTIAL: Reproduction phase (requires RNG and counting newborns)
        let allowed_newborns = self.max_preys.saturating_sub(self.preys.len());

        let mut newborn_preys: Vec<Prey> = Vec::new();

        for prey in self.preys.iter_mut() {
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
        if let Some(ref mut dm) = self.data_manager {
            dm.store_frame(&frame);
        }
    }

    pub async fn playback(file_name: &str, draw_sight_lines: bool) {
        // Build index for random access without loading all frames into memory
        println!("Indexing frames...");
        let mut frame_reader = match IndexedFrameReader::new(file_name) {
            Ok(reader) => reader,
            Err(e) => {
                eprintln!("Failed to open recording: {}", e);
                return;
            }
        };

        let total_frames = frame_reader.len();

        if total_frames == 0 {
            eprintln!("No frames found in recording.");
            return;
        }

        println!("Indexed {} frames. Starting playback...", total_frames);
        println!("Controls:");
        println!("  Space: Play/Pause");
        println!("  Left/Right Arrow: Step backward/forward");
        println!("  Up/Down Arrow: Increase/decrease speed");
        println!("  Home/End: Jump to start/end");
        println!("  Click and drag slider to seek");

        let mut playback_state = PlaybackState::default();
        let mut accumulated_time = 0.0;
        let frame_duration = 1.0 / 60.0; // Base frame rate

        // Cache the current frame to avoid re-reading on every render
        let mut cached_frame: Option<Frame> = None;
        let mut cached_frame_index: Option<usize> = None;

        loop {
            // Handle exit
            if is_key_pressed(KeyCode::Escape) || is_key_pressed(KeyCode::Q) {
                break;
            }

            // Update frame based on playback state
            if playback_state.is_playing && !playback_state.is_dragging {
                accumulated_time += get_frame_time() * playback_state.playback_speed;

                while accumulated_time >= frame_duration {
                    accumulated_time -= frame_duration;
                    if playback_state.current_frame < total_frames - 1 {
                        playback_state.current_frame += 1;
                    } else {
                        // Loop back to start or pause at end
                        playback_state.current_frame = 0;
                    }
                }
            }

            // Only read frame from disk if it changed
            if cached_frame_index != Some(playback_state.current_frame) {
                cached_frame = frame_reader.get_frame(playback_state.current_frame);
                cached_frame_index = Some(playback_state.current_frame);
            }

            let frame = match &cached_frame {
                Some(f) => f,
                None => {
                    eprintln!("Failed to read frame {}", playback_state.current_frame);
                    break;
                }
            };

            clear_background(settings::BACKGROUND_COLOR);
            draw_frame(frame, draw_sight_lines);

            let (pred_count, prey_count) = frame.counts();
            draw_game_stats(pred_count, prey_count, frame.tick);

            draw_playback_controls(&mut playback_state, total_frames);

            next_frame().await;
        }
    }
}
