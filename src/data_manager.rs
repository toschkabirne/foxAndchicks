use crate::animals::{Predator, Prey};
use crate::settings;
use bincode;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::SystemTime;

/// Snapshot of all simulation settings at the time of recording.
/// This captures both const and runtime-mutable settings.
/// All fields are `Option<T>` for forward compatibility - new fields added
/// in the future will be `None` when reading old files, clearly indicating
/// the setting was not available at recording time.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulationSettings {
    // Screen / game settings
    pub screen_width: Option<i32>,
    pub screen_height: Option<i32>,
    pub frames_per_second: Option<i32>,
    pub pred_init_numb: Option<usize>,
    pub prey_init_numb: Option<usize>,
    pub max_pred_count: Option<usize>,
    pub max_prey_count: Option<usize>,

    // Sight settings
    pub predator_sight_range: Option<f32>,
    pub prey_sight_range: Option<f32>,
    pub predator_sight_angle: Option<f32>,
    pub prey_sight_angle: Option<f32>,
    pub prey_sight_count: Option<usize>,
    pub predator_sight_count: Option<usize>,

    // Predator settings
    pub predator_radius: Option<f32>,
    pub predator_speed: Option<f32>,
    pub pred_energy: Option<f32>,
    pub predator_energy_gain: Option<f32>,
    pub pred_default_decay: Option<f32>,
    pub pred_moving_decay: Option<f32>,

    // Prey settings
    pub prey_radius: Option<f32>,
    pub prey_speed: Option<f32>,
    pub prey_energy: Option<f32>,
    pub prey_reproducation_rate: Option<f32>,
    pub prey_moving_decay: Option<f32>,
    pub prey_rest_energy_gain: Option<f32>,

    // Mutation parameters (runtime-mutable)
    pub prey_init_mut: Option<usize>,
    pub pred_init_mut: Option<usize>,
    pub add_neuron: Option<f32>,
    pub add_weight: Option<f32>,
    pub change_weight: Option<f32>,
    pub bias: Option<f32>,
}

impl SimulationSettings {
    /// Captures current settings from the settings module
    pub fn capture() -> Self {
        SimulationSettings {
            screen_width: Some(settings::SCREEN_WIDTH),
            screen_height: Some(settings::SCREEN_HEIGHT),
            frames_per_second: Some(settings::FRAMES_PER_SECOND),
            pred_init_numb: Some(settings::PRED_INIT_NUMB),
            prey_init_numb: Some(settings::PREY_INIT_NUMB),
            max_pred_count: Some(settings::MAX_PRED_COUNT),
            max_prey_count: Some(settings::MAX_PREY_COUNT),

            predator_sight_range: Some(settings::PRED_SIGHT_RANGE),
            prey_sight_range: Some(settings::PREY_SIGHT_RANGE),
            predator_sight_angle: Some(settings::PRED_SIGHT_ANGLE),
            prey_sight_angle: Some(settings::PREY_SIGHT_ANGLE),
            prey_sight_count: Some(settings::PREY_SIGHT_COUNT),
            predator_sight_count: Some(settings::PRED_SIGHT_COUNT),

            predator_radius: Some(settings::PRED_RADIUS),
            predator_speed: Some(settings::PRED_SPEED),
            pred_energy: Some(settings::PRED_ENERGY),
            predator_energy_gain: Some(settings::PRED_ENERGY_GAIN),
            pred_default_decay: Some(settings::PRED_DEFAULT_DECAY),
            pred_moving_decay: Some(settings::PRED_MOVING_DECAY),

            prey_radius: Some(settings::PREY_RADIUS),
            prey_speed: Some(settings::PREY_SPEED),
            prey_energy: Some(settings::PREY_ENERGY),
            prey_reproducation_rate: Some(settings::PREY_REPRODUCATION_RATE),
            prey_moving_decay: Some(settings::PREY_MOVING_DECAY),
            prey_rest_energy_gain: Some(settings::PREY_REST_ENERGY_GAIN),

            // Runtime-mutable settings
            prey_init_mut: Some(settings::prey_init_mut()),
            pred_init_mut: Some(settings::pred_init_mut()),
            add_neuron: Some(settings::add_neuron()),
            add_weight: Some(settings::add_weight()),
            change_weight: Some(settings::change_weight()),
            bias: Some(settings::bias()),
        }
    }
}

// Needed functionality
// we want to be able to track animals across frames

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AnimalType {
    Predator = 1,
    Prey = 0,
}

pub struct DataManager {
    // Fields and methods for managing data
    writer: BufWriter<File>,
    /// The actual filename used (including timestamp)
    pub filename: String,
}

impl DataManager {
    /// Creates a new DataManager for storing simulation data.
    /// Automatically appends a timestamp (YYYY-MM-DD_HH-MM-SS) to the filename.
    /// Settings are stored in a separate JSON file for forward/backward compatibility.
    ///
    /// Example: `new("simulation")` creates `simulation_2026-01-17_14-30-45.bin`
    pub fn new(base_filename: &str) -> Self {
        let filename = Self::generate_timestamped_filename(base_filename);
        let file = File::create(&filename).expect("Unable to create file");
        let writer = BufWriter::new(file);

        // Store settings in a separate JSON file for compatibility
        let settings = SimulationSettings::capture();
        let settings_filename = Self::settings_filename(&filename);
        let settings_json =
            serde_json::to_string_pretty(&settings).expect("Failed to serialize settings to JSON");
        let mut settings_file =
            File::create(&settings_filename).expect("Unable to create settings file");
        settings_file
            .write_all(settings_json.as_bytes())
            .expect("Failed to write settings file");

        DataManager { writer, filename }
    }

    /// Generates a filename with timestamp in the simulations folder.
    /// Format: `simulations/<base>_<unix_timestamp>.bin`
    fn generate_timestamped_filename(base: &str) -> String {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();

        let path = Path::new(base);
        if path.is_absolute() {
            let parent = path.parent().unwrap_or(Path::new(""));
            let file_stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or(base);
            format!("{}/{}_{}.bin", parent.display(), file_stem, timestamp)
        } else {
            let base_name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(base)
                .trim_end_matches(".bin");

            // Ensure simulations directory exists
            let simulations_dir = "simulations";
            fs::create_dir_all(simulations_dir).expect("Failed to create simulations directory");

            format!("{}/{}_{}.bin", simulations_dir, base_name, timestamp)
        }
    }

    /// Returns the settings filename for a given data filename.
    pub fn settings_filename(data_filename: &str) -> String {
        format!("{}.settings.json", data_filename.trim_end_matches(".bin"))
    }

    pub fn store_frame(&mut self, frame: &Frame) {
        bincode::serialize_into(&mut self.writer, frame).expect("Failed to write frame");
    }

    /// Reads the settings from the companion JSON file.
    /// Uses serde defaults for any missing fields (forward compatibility).
    pub fn read_settings(data_filename: &str) -> SimulationSettings {
        let settings_filename = Self::settings_filename(data_filename);
        let file = File::open(&settings_filename).expect("Unable to open settings file");
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).expect("Failed to parse settings JSON")
    }

    /// Returns an iterator that streams frames from a file using buffered reading.
    /// Each call to `next()` deserializes and returns the next frame.
    pub fn read_frames(filename: &str) -> FrameReader {
        let file = File::open(filename).expect("Unable to open file");
        let reader = BufReader::new(file);
        FrameReader { reader }
    }

    /// Returns both the settings and a frame iterator from a data file.
    pub fn read_file(filename: &str) -> (SimulationSettings, FrameReader) {
        let settings = Self::read_settings(filename);
        let frame_reader = Self::read_frames(filename);
        (settings, frame_reader)
    }
}

/// Iterator that streams frames from a binary file using buffered reading.
pub struct FrameReader {
    reader: BufReader<File>,
}

impl Iterator for FrameReader {
    type Item = Frame;

    fn next(&mut self) -> Option<Self::Item> {
        bincode::deserialize_from(&mut self.reader).ok()
    }
}

/// Random-access frame reader that builds an index of frame offsets.
/// This allows seeking to any frame without loading all frames into memory.
/// Only the index (8 bytes per frame) is kept in memory.
pub struct IndexedFrameReader {
    file: File,
    frame_offsets: Vec<u64>,
    current_index: usize,
}

impl IndexedFrameReader {
    /// Creates a new IndexedFrameReader by scanning the file to build an index.
    /// This performs one full read of the file to find frame boundaries.
    pub fn new(filename: &str) -> std::io::Result<Self> {
        // First pass: build the index
        let mut file = File::open(filename)?;
        let mut frame_offsets = Vec::new();
        let mut reader = BufReader::new(&file);

        loop {
            let offset = reader.stream_position()?;

            // Try to deserialize a frame to find its size
            let result: Result<Frame, _> = bincode::deserialize_from(&mut reader);
            match result {
                Ok(_) => {
                    frame_offsets.push(offset);
                }
                Err(_) => break, // End of file or parse error
            }
        }

        // Reopen file for random access (without BufReader for seeking)
        drop(reader);
        file.seek(SeekFrom::Start(0))?;

        Ok(Self {
            file,
            frame_offsets,
            current_index: 0,
        })
    }

    /// Returns the total number of frames in the file.
    pub fn len(&self) -> usize {
        self.frame_offsets.len()
    }

    /// Returns true if there are no frames.
    pub fn is_empty(&self) -> bool {
        self.frame_offsets.is_empty()
    }

    /// Seeks to and reads a specific frame by index.
    /// Returns None if the index is out of bounds.
    pub fn get_frame(&mut self, index: usize) -> Option<Frame> {
        if index >= self.frame_offsets.len() {
            return None;
        }

        let offset = self.frame_offsets[index];
        self.file.seek(SeekFrom::Start(offset)).ok()?;
        self.current_index = index;

        let mut reader = BufReader::new(&self.file);
        bincode::deserialize_from(&mut reader).ok()
    }

    /// Returns the current frame index.
    pub fn current_index(&self) -> usize {
        self.current_index
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Frame {
    pub tick: usize,
    pub animals: Vec<AnimalState>,
}

impl Frame {
    pub fn new(predators: &Vec<Predator>, preys: &Vec<Prey>, tick: usize) -> Self {
        let mut animal_states = Vec::with_capacity(predators.len() + preys.len());

        for p in predators.iter() {
            animal_states.push(AnimalState {
                id: p.core.id,
                x: p.core.pos.x,
                y: p.core.pos.y,
                angle: p.core.angle,
                animal_type: AnimalType::Predator,
            });
        }

        for p in preys.iter() {
            animal_states.push(AnimalState {
                id: p.core.id,
                x: p.core.pos.x,
                y: p.core.pos.y,
                angle: p.core.angle,
                animal_type: AnimalType::Prey,
            });
        }

        Frame {
            tick,
            animals: animal_states,
        }
    }

    /// Returns the count of predators and preys in this frame
    pub fn counts(&self) -> (usize, usize) {
        let pred_count = self
            .animals
            .iter()
            .filter(|a| a.animal_type == AnimalType::Predator)
            .count();
        let prey_count = self
            .animals
            .iter()
            .filter(|a| a.animal_type == AnimalType::Prey)
            .count();
        (pred_count, prey_count)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AnimalState {
    pub id: usize,
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub animal_type: AnimalType,
}

////////////////////////
// TESTS
////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use ::rand::rngs::StdRng;
    use ::rand::SeedableRng;

    #[test]
    fn test_animal_state_creation() {
        let state = AnimalState {
            id: 1,
            x: 10.0,
            y: 20.0,
            angle: 45.0,
            animal_type: AnimalType::Predator,
        };
        assert_eq!(state.id, 1);
        assert_eq!(state.x, 10.0);
        assert_eq!(state.y, 20.0);
    }

    #[test]
    fn test_frame_creation_empty() {
        let predators: Vec<Predator> = Vec::new();
        let preys: Vec<Prey> = Vec::new();
        let frame = Frame::new(&predators, &preys, 0);
        assert_eq!(frame.animals.len(), 0);
    }

    #[test]
    fn test_frame_creation_with_animals() {
        let mut rng = StdRng::seed_from_u64(settings::SEED);
        let predator = Predator::new(10.0, 20.0, &mut rng);
        let prey = Prey::new(30.0, 40.0, &mut rng);

        let predators = vec![predator];
        let preys = vec![prey];
        let frame = Frame::new(&predators, &preys, 0);

        assert_eq!(frame.animals.len(), 2);
        // IDs should be unique, but not necessarily 1 and 2 due to parallel tests
        assert_ne!(frame.animals[0].id, frame.animals[1].id);
    }

    #[test]
    fn test_store_frame() {
        let mut dm = DataManager::new("/tmp/test_store_frame");
        let actual_filename = dm.filename.clone();
        let predators: Vec<Predator> = Vec::new();
        let preys: Vec<Prey> = Vec::new();
        let frame = Frame::new(&predators, &preys, 0);

        dm.store_frame(&frame);
        drop(dm); // Ensure file is flushed

        // Verify the timestamped file was created
        assert!(std::path::Path::new(&actual_filename).exists());
        assert!(actual_filename.contains("test_store_frame_"));
        assert!(actual_filename.ends_with(".bin"));

        // Verify we can read the settings and frames back
        let (settings, mut frame_reader) = DataManager::read_file(&actual_filename);
        assert_eq!(settings.screen_width, Some(crate::settings::SCREEN_WIDTH));
        assert_eq!(settings.screen_height, Some(crate::settings::SCREEN_HEIGHT));
        assert_eq!(
            settings.predator_sight_range,
            Some(crate::settings::PRED_SIGHT_RANGE)
        );

        let read_frame = frame_reader.next().expect("Should have one frame");
        assert_eq!(read_frame.tick, 0);
        assert_eq!(read_frame.animals.len(), 0);

        // Verify settings JSON file was created
        let settings_path = DataManager::settings_filename(&actual_filename);
        assert!(std::path::Path::new(&settings_path).exists());

        // Cleanup
        let _ = std::fs::remove_file(&actual_filename);
        let _ = std::fs::remove_file(&settings_path);
    }

    #[test]
    fn test_settings_storage() {
        let dm = DataManager::new("/tmp/test_settings_storage");
        let actual_filename = dm.filename.clone();
        drop(dm); // Ensure file is flushed

        // Verify settings JSON file was created with timestamp
        let settings_path = DataManager::settings_filename(&actual_filename);
        assert!(std::path::Path::new(&settings_path).exists());

        let settings = DataManager::read_settings(&actual_filename);
        assert_eq!(settings.screen_width, Some(crate::settings::SCREEN_WIDTH));
        assert_eq!(
            settings.frames_per_second,
            Some(crate::settings::FRAMES_PER_SECOND)
        );
        assert_eq!(
            settings.pred_init_numb,
            Some(crate::settings::PRED_INIT_NUMB)
        );
        assert_eq!(
            settings.prey_init_numb,
            Some(crate::settings::PREY_INIT_NUMB)
        );

        // Cleanup
        let _ = std::fs::remove_file(&actual_filename);
        let _ = std::fs::remove_file(&settings_path);
    }
}
