
/*
Contains the implementation of Animals (Prey and Predator) with their neural networks, movement, and mutation logic.
*/


use rand::RngCore;
use rand::Rng;
use rand::prelude::IndexedRandom;
use std::ops::Index;
use neural_network::{NeuralNetwork, LayerTopology};
use std::ops::{Deref, DerefMut};



pub const TIME_STEP: f32 = 0.1;
pub const BASE_FITNESS: usize = 5;
pub const MUTATION_RATE: f32 = 0.1;
pub const MUTATION_AMOUNT: f32 = 0.5;

#[derive(Debug)]
pub struct Animal {
    pub id: usize,
    pub pos: (f32, f32),
    pub sight: Vec<f32>,
    pub speed: f32,
    pub angle: f32,
    pub fitness: usize,
    pub brain: NeuralNetwork,
}

impl Animal {
    pub fn new(id: usize, pos: (f32, f32), sight: Vec<f32>, brain_topology: &[LayerTopology]) -> Self {
        
        let brain = NeuralNetwork::new(brain_topology);
        let out = brain.propagate(sight.clone());
        let speed = out[0];
        let angle = out[1];
        Self { id, pos, sight, speed, angle, fitness: BASE_FITNESS, brain }
    }

    pub fn update_sight(&mut self, new_sight: Vec<f32>) {
        self.sight = new_sight;
        let out = self.brain.propagate(new_sight);
        self.speed = out[0];
        self.angle = out[1];
    }

    pub fn step(&mut self) {
        self.pos.0 += self.speed * TIME_STEP * self.angle.cos();
        self.pos.1 += self.speed * TIME_STEP * self.angle.sin();
    }

    pub fn mutate(&mut self) {
        self.brain.mutate(MUTATION_RATE, MUTATION_AMOUNT);
    }

}

impl Clone for Animal {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            pos: self.pos,
            sight: self.sight.clone(),
            speed: self.speed,
            angle: self.angle,
            fitness: self.fitness,
            brain: self.brain.clone(),
        }
    }
}

#[derive(Debug)]
pub struct Prey(Animal);
pub struct Predator(Animal);



impl Deref for Prey {
    type Target = Animal;

    fn deref(&self) -> &Self::Target {
        &self.animal
    }
}

impl DerefMut for Prey {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.animal
    }
}

impl Deref for Predator {
    type Target = Animal;

    fn deref(&self) -> &Self::Target {
        &self.animal
    }
}

impl DerefMut for Predator {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.animal
    }
}


impl Prey {
    pub fn new(id: usize, pos: (f32, f32), sight: Vec<f32>, brain_topology: &[LayerTopology]) -> Self {
        let animal = Animal::new(id, pos, sight, brain_topology);
        Self(animal)
    }
}


impl Predator {
    pub fn new(id: usize, pos: (f32, f32), sight: Vec<f32>, brain_topology: &[LayerTopology]) -> Self {
        let animal = Animal::new(id, pos, sight, brain_topology);
        Self(animal)
    }
}