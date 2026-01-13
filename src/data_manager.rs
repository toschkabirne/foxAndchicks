use crate::animals::{Predator, Prey};
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, BufReader};
use std::rc::Rc;
use serde::{Serialize, Deserialize};
use bincode;

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
}


impl DataManager {

    pub fn new(filename: &str) -> Self {
        let file = File::create(&filename).expect("Unable to create file");
        let writer = BufWriter::new(file);
        DataManager {
            writer,
        }
    }

    pub fn store_frame(&mut self, frame: &Frame) {
        bincode::serialize_into(&mut self.writer, frame).expect("Failed to write frame");
    }

    /// Returns an iterator that streams frames from a file using buffered reading.
    /// Each call to `next()` deserializes and returns the next frame.
    pub fn read_frames(filename: &str) -> FrameReader {
        let file = File::open(filename).expect("Unable to open file");
        let reader = BufReader::new(file);
        FrameReader { reader }
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Frame {
    pub tick: usize,
    pub animals: Vec<AnimalState>,
}

impl Frame {
    pub fn new(predators: &Vec<Rc<RefCell<Predator>>>, preys: &Vec<Rc<RefCell<Prey>>>, tick: usize) -> Self {
        let mut animal_states = Vec::new();

        for p in predators.iter() {
            let p_borrow = p.borrow();
            animal_states.push(AnimalState {
                id: p_borrow.id,
                x: p_borrow.x,
                y: p_borrow.y,
                angle: p_borrow.angle,
                animal_type: AnimalType::Predator,
            });
        }

        for p in preys.iter() {
            let p_borrow = p.borrow();
            animal_states.push(AnimalState {
                id: p_borrow.id,
                x: p_borrow.x,
                y: p_borrow.y,
                angle: p_borrow.angle,
                animal_type: AnimalType::Prey,
            });
        }

        Frame {
            tick: tick,
            animals: animal_states,
        }
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
    use rand::rngs::ThreadRng;

    use super::*;

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
        let predators: Vec<Rc<RefCell<Predator>>> = Vec::new();
        let preys: Vec<Rc<RefCell<Prey>>> = Vec::new();
        let frame = Frame::new(&predators, &preys, 0);
        assert_eq!(frame.animals.len(), 0);
    }

    #[test]
    fn test_frame_creation_with_animals() {
        let mut rng: ThreadRng = ::rand::thread_rng();
        let predator = Rc::new(RefCell::new(Predator::new(10.0, 20.0, &mut rng)));
        let prey = Rc::new(RefCell::new(Prey::new(30.0, 40.0, &mut rng)));
        
        let predators = vec![predator];
        let preys = vec![prey];
        let frame = Frame::new(&predators, &preys, 0);
        
        assert_eq!(frame.animals.len(), 2);
        assert_eq!(frame.animals[0].id, 1);
        assert_eq!(frame.animals[1].id, 2);
    }

    #[test]
    fn test_store_frame() {
        let filename = "/tmp/test_store_frame.bin";
        let mut dm = DataManager::new(filename);
        let predators: Vec<Rc<RefCell<Predator>>> = Vec::new();
        let preys: Vec<Rc<RefCell<Prey>>> = Vec::new();
        let frame = Frame::new(&predators, &preys, 0);
        
        dm.store_frame(&frame);
        assert!(std::path::Path::new(filename).exists());
    }
}
