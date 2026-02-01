// Benchmark run to test impact of spatial hash on performance
// Runs with different world sizes and numbers of predators and prey
// Compares spatial hash approach with naive approach
// Run with: cargo run --bin benchmark

use predator_vs_prey::animals::{normalize_angle, wrapped_distance_vector, Predator, Prey};
use predator_vs_prey::settings::{self};
use predator_vs_prey::spatial_hash::SpatialHash;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

fn main() {
    if std::path::Path::new("benchmark_results.csv").exists() {
        println!(
            "Benchmark file already exists. Delete it if you want to run the benchmark again."
        );
        return;
    }
    let mut file = File::create("benchmark_results.csv").expect("Could not create benchmark file");
    let header = "Benchmarking Spatial Hash vs Naive Approach (Average of 10 runs)";
    println!("{}", header);
    // Write CSV header
    writeln!(
        file,
        "world_size,num_preds,num_preys,iteration,phase,sh_ms,naive_ms,speedup"
    )
    .unwrap();
    println!("--------------------------------------------------");

    let scenarios = vec![
        (10, 10),
        (50, 50),
        (100, 100),
        (500, 500),
        (1000, 1000),
        (2000, 2000),
        (4000, 4000),
    ];

    //let world_sizes = vec![4000.0];
    let world_sizes = vec![600.0, 1200.0, 2000.0, 4000.0];

    for world_size in world_sizes {
        settings::set_screen_width(world_size as i32);
        settings::set_screen_height(world_size as i32);
        settings::set_pred_sight_range(200.0);
        settings::set_prey_sight_range(200.0);
        settings::set_pred_sight_angle(60.0);
        settings::set_prey_sight_angle(300.0);
        println!("\n==================================================");
        println!("WORLD SIZE: {}x{}", world_size, world_size);
        println!("==================================================");
        for (num_preds, num_preys) in &scenarios {
            run_benchmark(*num_preds, *num_preys, world_size, &mut file);
        }
    }
}

fn run_benchmark(num_preds: usize, num_preys: usize, world_size: f32, file: &mut File) {
    println!("\nScenario: {} Predators vs {} Preys", num_preds, num_preys);

    let iterations = 10;
    let mut total_sh_pred = Duration::ZERO;
    let mut total_naive_pred = Duration::ZERO;
    let mut total_sh_prey = Duration::ZERO;
    let mut total_naive_prey = Duration::ZERO;

    let mut neighbors_sh_pred = 0;
    let mut neighbors_naive_pred = 0;
    let mut neighbors_sh_prey = 0;
    let mut neighbors_naive_prey = 0;

    for i in 0..iterations {
        let mut rng = StdRng::seed_from_u64(42 + i as u64);

        let predators: Vec<Predator> = (0..num_preds)
            .map(|_| {
                Predator::new(
                    rng.gen_range(0.0..world_size),
                    rng.gen_range(0.0..world_size),
                    &mut rng,
                )
            })
            .collect();

        let preys: Vec<Prey> = (0..num_preys)
            .map(|_| {
                Prey::new(
                    rng.gen_range(0.0..world_size),
                    rng.gen_range(0.0..world_size),
                    &mut rng,
                )
            })
            .collect();

        let world_w = world_size;
        let world_h = world_size;

        // ----------------------------------------------------------------
        // Phase 1: Predators sensing Prey
        // ----------------------------------------------------------------
        let pred_sight_range = settings::pred_sight_range();
        let pred_sight_angle = settings::pred_sight_angle();
        let cell_size_prey = (pred_sight_range.ceil() as usize).max(1);
        let mut hash_prey = SpatialHash::new(cell_size_prey, world_w, world_h);

        let mut params_buffer = Vec::new();

        // SH
        let start_sh_pred = Instant::now();
        hash_prey.rebuild_from(&preys);
        for pred in &predators {
            hash_prey.query_into(&mut params_buffer, pred.core.x(), pred.core.y());
            for &idx in &params_buffer {
                let prey = &preys[idx];
                let delta = wrapped_distance_vector(pred.core.pos, prey.core.pos, world_w, world_h);
                let dist = delta.length();
                if dist <= pred_sight_range && dist > 0.0 {
                    let angle_to_prey = delta.y.atan2(delta.x);
                    let half_fov = (pred_sight_angle / 2.0).to_radians();
                    let diff = normalize_angle(angle_to_prey - pred.core.angle).abs();
                    if diff <= half_fov {
                        if i == 0 {
                            neighbors_sh_pred += 1;
                        }
                    }
                }
            }
        }
        let elapsed_sh_pred = start_sh_pred.elapsed();
        total_sh_pred += elapsed_sh_pred;

        // Naive
        let start_naive_pred = Instant::now();
        for pred in &predators {
            for prey in &preys {
                let delta = wrapped_distance_vector(pred.core.pos, prey.core.pos, world_w, world_h);
                let dist = delta.length();
                if dist <= pred_sight_range && dist > 0.0 {
                    let angle_to_prey = delta.y.atan2(delta.x);
                    let half_fov = (pred_sight_angle / 2.0).to_radians();
                    let diff = normalize_angle(angle_to_prey - pred.core.angle).abs();
                    if diff <= half_fov {
                        if i == 0 {
                            neighbors_naive_pred += 1;
                        }
                    }
                }
            }
        }
        let elapsed_naive_pred = start_naive_pred.elapsed();
        total_naive_pred += elapsed_naive_pred;

        // ----------------------------------------------------------------
        // Phase 2: Prey sensing Predators
        // ----------------------------------------------------------------
        let prey_sight_range = settings::prey_sight_range();
        let prey_sight_angle = settings::prey_sight_angle();
        let cell_size_pred = (prey_sight_range.ceil() as usize).max(1);
        let mut hash_pred = SpatialHash::new(cell_size_pred, world_w, world_h);

        // SH
        let start_sh_prey = Instant::now();
        hash_pred.rebuild_from(&predators);
        for prey in &preys {
            hash_pred.query_into(&mut params_buffer, prey.core.x(), prey.core.y());
            for &idx in &params_buffer {
                let pred = &predators[idx];
                let delta = wrapped_distance_vector(prey.core.pos, pred.core.pos, world_w, world_h);
                let dist = delta.length();
                if dist <= prey_sight_range && dist > 0.0 {
                    let angle_to_pred = delta.y.atan2(delta.x);
                    let half_fov = (prey_sight_angle / 2.0).to_radians();
                    let diff = normalize_angle(angle_to_pred - prey.core.angle).abs();
                    if diff <= half_fov {
                        if i == 0 {
                            neighbors_sh_prey += 1;
                        }
                    }
                }
            }
        }
        let elapsed_sh_prey = start_sh_prey.elapsed();
        total_sh_prey += elapsed_sh_prey;

        // Naive
        let start_naive_prey = Instant::now();
        for prey in &preys {
            for pred in &predators {
                let delta = wrapped_distance_vector(prey.core.pos, pred.core.pos, world_w, world_h);
                let dist = delta.length();
                if dist <= prey_sight_range && dist > 0.0 {
                    let angle_to_pred = delta.y.atan2(delta.x);
                    let half_fov = (prey_sight_angle / 2.0).to_radians();
                    let diff = normalize_angle(angle_to_pred - prey.core.angle).abs();
                    if diff <= half_fov {
                        if i == 0 {
                            neighbors_naive_prey += 1;
                        }
                    }
                }
            }
        }
        let elapsed_naive_prey = start_naive_prey.elapsed();
        total_naive_prey += elapsed_naive_prey;

        // Write CSV rows
        let total_sh = elapsed_sh_pred + elapsed_sh_prey;
        let total_naive = elapsed_naive_pred + elapsed_naive_prey;

        writeln!(
            file,
            "{},{},{},{},combined,{:.3},{:.3},{:.2}",
            world_size,
            num_preds,
            num_preys,
            i,
            total_sh.as_secs_f64() * 1000.0,
            total_naive.as_secs_f64() * 1000.0,
            total_naive.as_secs_f64() / total_sh.as_secs_f64().max(1e-9)
        )
        .unwrap();
    }

    let avg_sh_pred = total_sh_pred / iterations;
    let avg_naive_pred = total_naive_pred / iterations;
    let avg_sh_prey = total_sh_prey / iterations;
    let avg_naive_prey = total_naive_prey / iterations;

    println!("  [Predator sensing Prey]");
    println!(
        "    -> Neighbors found: SH={} Naive={}",
        neighbors_sh_pred, neighbors_naive_pred
    );
    println!("    -> Avg. SH Time: {:.3?}", avg_sh_pred);
    println!("    -> Avg. Naive Time: {:.3?}", avg_naive_pred);
    println!(
        "    -> Avg. Speedup: {:.2}x",
        avg_naive_pred.as_secs_f64() / avg_sh_pred.as_secs_f64().max(1e-9)
    );

    println!("  [Prey sensing Predator]");
    println!(
        "    -> Neighbors found: SH={} Naive={}",
        neighbors_sh_prey, neighbors_naive_prey
    );
    println!("    -> Avg. SH Time: {:.3?}", avg_sh_prey);
    println!("    -> Avg. Naive Time: {:.3?}", avg_naive_prey);
    println!(
        "    -> Avg. Speedup: {:.2}x",
        avg_naive_prey.as_secs_f64() / avg_sh_prey.as_secs_f64().max(1e-9)
    );
}
