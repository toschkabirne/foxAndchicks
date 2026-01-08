// src/main.rs

use predatorVsPrey::animals::{Predator, Prey};
use predatorVsPrey::settings;
use predatorVsPrey::spatial_hash::SpatialHash;

use ::rand::rngs::ThreadRng;
use ::rand::Rng;
use macroquad::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

fn window_conf() -> Conf {
    Conf {
        window_title: "Predator and Prey Simulation".to_string(),
        window_width: settings::SCREEN_WIDTH,
        window_height: settings::SCREEN_HEIGHT,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    // macroquad sets target fps implicitly via frame time, but let's see if we can just cap it or rely on vsync.
    // Rust macroquad doesn't have set_target_fps exposed directly in prelude in recent versions?
    // Checking docs: 'set_target_fps' is not in standard macroquad 0.4 prelude?
    // Actually, usually users just use window conf or await next_frame().
    // But the error said `set_target_fps` not found.
    // We can try to remove it or find where it is.
    // For now I will comment it out as it is not critical for basic wiring, or use a proper alternative if I knew one.
    // However, I will try to leave it out for now to satisfy compilation.
    // set_target_fps(settings::FRAMES_PER_SECOND as u32);

    let mut rng: ThreadRng = ::rand::thread_rng();

    let mut predators: Vec<Rc<RefCell<Predator>>> = (0..settings::PRED_INIT_NUMB)
        .map(|_| {
            Rc::new(RefCell::new(Predator::new(
                rng.gen_range(0.0..settings::SCREEN_WIDTH as f32),
                rng.gen_range(0.0..settings::SCREEN_HEIGHT as f32),
                &mut rng,
            )))
        })
        .collect();

    let mut preys: Vec<Rc<RefCell<Prey>>> = (0..settings::PREY_INIT_NUMB)
        .map(|_| {
            Rc::new(RefCell::new(Prey::new(
                rng.gen_range(0.0..settings::SCREEN_WIDTH as f32),
                rng.gen_range(0.0..settings::SCREEN_HEIGHT as f32),
                &mut rng,
            )))
        })
        .collect();

    // Python: SpatialHash(SCREEN_WIDTH // SIGHT_RANGE_*)
    let cell_pred =
        ((settings::SCREEN_WIDTH as f32) / settings::SIGHT_RANGE_PREDATOR).floor() as i32;
    let cell_prey = ((settings::SCREEN_WIDTH as f32) / settings::SIGHT_RANGE_PREY).floor() as i32;

    let mut spatial_preds: SpatialHash<Predator> = SpatialHash::new(cell_pred);
    let mut spatial_preys: SpatialHash<Prey> = SpatialHash::new(cell_prey);

    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        clear_background(BLACK);

        // insert all predators + preys (like Python)
        for p in &predators {
            spatial_preds.insert(Rc::clone(p));
        }
        for pr in &preys {
            spatial_preys.insert(Rc::clone(pr));
        }

        // --- Predator update/draw ---
        // Python removes while iterating; we emulate that semantics safely by storing Rc in spatial hash.
        let mut i = 0;
        while i < predators.len() {
            let pred_rc = Rc::clone(&predators[i]);
            let (px, py) = {
                let pred = pred_rc.borrow();
                (pred.x, pred.y)
            };

            let nearby_preys = spatial_preys.query(px, py);

            let dead = {
                let mut pred = pred_rc.borrow_mut();
                let inputs = pred.get_inputs(&nearby_preys);
                pred.move_step(&inputs);
                pred.energy < 0.0
            };

            // draw (even if removed from list this frame, like Python loop variable still exists)
            pred_rc.borrow().draw();

            if dead {
                predators.remove(i);
            } else {
                i += 1;
            }
        }

        // --- Prey update/draw ---
        let mut j = 0;
        while j < preys.len() {
            let prey_rc = Rc::clone(&preys[j]);
            let (x, y) = {
                let prey = prey_rc.borrow();
                (prey.x, prey.y)
            };

            let nearby_preds = spatial_preds.query(x, y);

            // Python: inputs = prey.get_inputs(...); if inputs: move else remove
            let inputs_opt = {
                let prey = prey_rc.borrow();
                prey.get_inputs(&nearby_preds, &mut predators, &mut rng)
            };

            let Some(inputs) = inputs_opt else {
                preys.remove(j);
                continue;
            };

            {
                let mut prey = prey_rc.borrow_mut();
                prey.move_step(&inputs);

                if preys.len() < 900 {
                    if let Some(new_prey) = prey.reproduce(&mut rng) {
                        preys.push(new_prey);
                    }
                }
            }

            prey_rc.borrow().draw();

            j += 1;
        }

        // clear spatial hashes each frame (like Python)
        spatial_preds.clear();
        spatial_preys.clear();

        next_frame().await;
    }
}
