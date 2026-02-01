use crate::animals::{wrapped_distance_abs, Predator, Prey};
use crate::data_manager::{AnimalType, DataManager, Frame, IndexedFrameReader};
use crate::settings::{self};
use crate::spatial_hash::SpatialHash;
use crate::visualization::{
    draw_frame, draw_game_stats, draw_playback_controls, draw_population_graph_fullscreen,
    graph_enabled, PlaybackState,
};

use ::rand::rngs::StdRng;
use ::rand::{Rng, SeedableRng};
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
    pub data_manager: Option<DataManager>,
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

    /// Finds the animal closest to the given position (x, y) within a small radius.
    /// Returns Some((AnimalType, id)) if found, None otherwise.
    pub fn get_closest_animal_at(&self, x: f32, y: f32) -> Option<(AnimalType, usize)> {
        let click_pos = Vec2::new(x, y);
        let selection_radius = 40.0; // Reasonable click radius
        let world_w = settings::screen_width() as f32;
        let world_h = settings::screen_height() as f32;

        let mut closest_d = selection_radius;
        let mut found = None;

        // Check predators
        for pred in &self.predators {
            let dist = wrapped_distance_abs(click_pos, pred.core.pos, world_w, world_h);
            if dist < closest_d {
                closest_d = dist;
                found = Some((AnimalType::Predator, pred.core.id));
            }
        }

        // Check prey (if closer than any predator found so far)
        for prey in &self.preys {
            let dist = wrapped_distance_abs(click_pos, prey.core.pos, world_w, world_h);
            if dist < closest_d {
                closest_d = dist;
                found = Some((AnimalType::Prey, prey.core.id));
            }
        }

        found
    }

    /// Retrieves the brain of the specified animal.
    pub fn get_animal_brain(
        &self,
        animal_type: AnimalType,
        id: usize,
    ) -> Option<&crate::brain_neural_network::NeuralNetwork> {
        match animal_type {
            AnimalType::Predator => self
                .predators
                .iter()
                .find(|p| p.core.id == id)
                .map(|p| &p.core.brain),
            AnimalType::Prey => self
                .preys
                .iter()
                .find(|p| p.core.id == id)
                .map(|p| &p.core.brain),
        }
    }

    /// Creates a new Game with custom parameters
    pub fn new(
        file_name: Option<&str>,
        num_preds: usize,
        num_preys: usize,
        max_preds: usize,
        max_preys: usize,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(settings::SEED);

        // Spawn initial predators and preys as Rc<RefCell<>> for shared mutability
        let predators: Vec<Predator> = (0..num_preds)
            .map(|_| {
                Predator::new(
                    rng.gen_range(0.0..settings::screen_width() as f32),
                    rng.gen_range(0.0..settings::screen_height() as f32),
                    &mut rng,
                )
            })
            .collect();

        let preys: Vec<Prey> = (0..num_preys)
            .map(|_| {
                Prey::new(
                    rng.gen_range(0.0..settings::screen_width() as f32),
                    rng.gen_range(0.0..settings::screen_height() as f32),
                    &mut rng,
                )
            })
            .collect();

        // Set up spatial hash
        // Predators query prey_hash, so prey_hash cell size should be PRED_SIGHT_RANGE
        let cell_size_prey = settings::pred_sight_range() as usize;
        // Preys query pred_hash, so pred_hash cell size should be PREY_SIGHT_RANGE
        let cell_size_pred = settings::prey_sight_range() as usize;

        let world_w = settings::screen_width() as f32;
        let world_h = settings::screen_height() as f32;

        let spatial_hash_preds: SpatialHash = SpatialHash::new(cell_size_pred, world_w, world_h);
        let spatial_hash_preys: SpatialHash = SpatialHash::new(cell_size_prey, world_w, world_h);

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

                pred.move_step(&inputs);
            },
        );

        // SEQUENTIAL: Hunting phase (requires mutable shared state for eaten_prey_ids)
        let mut eaten_prey_ids: HashSet<usize> = HashSet::new();
        let mut newborn_preds: Vec<Predator> = Vec::new();
        let mut rng = StdRng::seed_from_u64(settings::SEED);

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
        self.predators.retain(|pred| pred.core.energy > 0.0);

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

                prey.move_step(&inputs);
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

        // only build history if the user asked for it (avoids rereading the file otherwise).
        let pop_history: Vec<(usize, usize)> = if graph_enabled() {
            let mut h = Vec::with_capacity(total_frames);
            for f in DataManager::read_frames(file_name) {
                h.push(f.counts());
            }
            h
        } else {
            Vec::new()
        };

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum ViewMode {
            Simulation,
            Graph,
        }

        let mut view_mode = ViewMode::Simulation;

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

            // --- INPUT / CONTROL PHASE ---
            if graph_enabled() && is_key_pressed(KeyCode::G) {
                view_mode = match view_mode {
                    ViewMode::Simulation => ViewMode::Graph,
                    ViewMode::Graph => ViewMode::Simulation,
                };
            }

            // Update frame based on playback state
            if playback_state.is_playing && !playback_state.is_dragging {
                accumulated_time += get_frame_time() * playback_state.playback_speed;

                while accumulated_time >= frame_duration {
                    accumulated_time -= frame_duration;
                    if playback_state.current_frame < total_frames - 1 {
                        playback_state.current_frame += 1;
                    } else {
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

            // --- DRAW PHASE ---
            match view_mode {
                ViewMode::Simulation => {
                    draw_frame(frame, draw_sight_lines, None);

                    let (pred_count, prey_count) = frame.counts();
                    draw_game_stats(pred_count, prey_count, frame.tick);

                    if graph_enabled() {
                        draw_text(
                            "Press G for graph view",
                            15.0,
                            26.0,
                            22.0,
                            Color::from_rgba(220, 220, 220, 255),
                        );
                    }
                }

                ViewMode::Graph => {
                    // Fullscreen graph view
                    draw_population_graph_fullscreen(
                        &pop_history,
                        playback_state.current_frame,
                        total_frames,
                    );

                    let (pred_count, prey_count) = frame.counts();
                    draw_game_stats(pred_count, prey_count, frame.tick);
                }
            }

            // Controls visible in both modes
            draw_playback_controls(&mut playback_state, total_frames);

            next_frame().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_hash_fov_inclusion() {
        // Setup: Predator at (500, 500), Prey at (500 + 100, 500)
        // Sight range 170. Prey is within range.
        let mut game = Game::new(None, 1, 1, 10, 10);

        // Move predator to center
        game.predators[0].core.set_xy(500.0, 500.0);

        // Move prey to 1 pixel within sight range
        let sight_range = settings::pred_sight_range();
        game.preys[0].core.set_xy(500.0 + sight_range - 1.0, 500.0);

        // Rebuild hash
        game.spatial_hash_preys.rebuild_from(&game.preys);

        // Query neighbors
        let nearby_prey_indices = game.spatial_hash_preys.query(500.0, 500.0);

        // Assert: Prey should be found
        assert!(
            !nearby_prey_indices.is_empty(),
            "Prey within FOV was not detected by SpatialHash"
        );
        assert_eq!(nearby_prey_indices[0], 0);
    }

    #[test]
    fn test_vision_range_exclusion() {
        // Setup: Predator at (500, 500), Prey at (500 + 2*sight_range + 10, 500)
        // Sight range 150. Prey is OUTSIDE sight range.
        // However, it might still be in a neighboring spatial hash cell.
        // We want to verify that the predator's fine-grained sensing logic
        // correctly excludes it even if the spatial hash returns it as a candidate.

        let mut game = Game::new(None, 1, 1, 10, 10);

        // Move predator to center
        game.predators[0].core.set_xy(500.0, 500.0);

        // Move prey to well outside sight range
        let sight_range = settings::pred_sight_range();
        game.preys[0]
            .core
            .set_xy(500.0 + 2.0 * sight_range + 10.0, 500.0);

        // Rebuild hash
        game.spatial_hash_preys.rebuild_from(&game.preys);

        // 1. Query neighbors (SpatialHash)
        let nearby_prey_indices = game.spatial_hash_preys.query(500.0, 500.0);

        // 2. Verify sensing logic EXCLUDES it (fine filter)
        // Even if the spatial hash is coarse and returns it, the predator shouldn't see it.
        let inputs = game.predators[0].get_inputs(
            nearby_prey_indices
                .iter()
                .filter_map(|&i| game.preys.get(i)),
        );

        let sum: f32 = inputs.iter().sum();
        assert_eq!(
            sum, 0.0,
            "Predator should NOT see the prey outside its vision range"
        );
    }

    #[test]
    fn test_vision_fov_exclusion() {
        // Setup: Predator at (500, 500) facing right (0.0 rad)
        // Prey at (550, 600).
        // Angle to prey: atan2(100, 50) = 1.107 rad (~63.4 deg)
        // Predator FOV is 90 deg (±45 deg).
        // The prey should be OUTSIDE FOV but INSIDE spatial hash neighborhood.

        let mut game = Game::new(None, 1, 1, 10, 10);

        // Facing East
        game.predators[0].core.set_xy(500.0, 500.0);
        game.predators[0].core.angle = 0.0;

        // Position where it's within spatial hash but outside FOV.
        // We place it at an angle slightly larger than half-FOV.
        let half_fov_rad = (settings::pred_sight_angle() / 2.0).to_radians();
        let exclusion_angle = half_fov_rad + 0.1;

        let dist = settings::pred_sight_range() * 0.5; // Well within range
        let prey_x = 500.0 + dist * exclusion_angle.cos();
        let prey_y = 500.0 + dist * exclusion_angle.sin();

        game.preys[0].core.set_xy(prey_x, prey_y);

        game.spatial_hash_preys.rebuild_from(&game.preys);

        // 1. Verify SpatialHash DOES find it (coarse filter)
        let nearby_indices = game.spatial_hash_preys.query(500.0, 500.0);
        assert!(
            !nearby_indices.is_empty(),
            "SpatialHash should return the prey as a candidate"
        );

        // 2. Verify sensing logic EXCLUDES it (fine filter)
        let inputs =
            game.predators[0].get_inputs(nearby_indices.iter().filter_map(|&i| game.preys.get(i)));

        let sum: f32 = inputs.iter().sum();
        assert_eq!(sum, 0.0, "Predator should NOT see the prey outside its FOV");
    }

    #[test]
    fn test_determinism_with_seed() {
        // Create two games with the same seed
        let mut game1 = Game::new(None, 5, 10, 50, 100);
        let mut game2 = Game::new(None, 5, 10, 50, 100);

        // Run 10 frames on both games
        for _ in 0..10 {
            game1.next_frame();
            game2.next_frame();
        }

        // Verify that both games have the same number of predators and prey
        assert_eq!(
            game1.predator_count(),
            game2.predator_count(),
            "Predator counts should be identical with same seed"
        );
        assert_eq!(
            game1.prey_count(),
            game2.prey_count(),
            "Prey counts should be identical with same seed"
        );

        // Verify that the first predator has the same position in both games
        if !game1.predators.is_empty() && !game2.predators.is_empty() {
            let pred1_pos = game1.predators[0].core.pos;
            let pred2_pos = game2.predators[0].core.pos;
            assert_eq!(
                pred1_pos, pred2_pos,
                "First predator position should be identical with same seed"
            );
        }

        // Verify that the first prey has the same position in both games
        if !game1.preys.is_empty() && !game2.preys.is_empty() {
            let prey1_pos = game1.preys[0].core.pos;
            let prey2_pos = game2.preys[0].core.pos;
            assert_eq!(
                prey1_pos, prey2_pos,
                "First prey position should be identical with same seed"
            );
        }
    }
}
