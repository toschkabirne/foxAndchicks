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

const WARMUP_RUNS: usize = 10; // number of unrecorded warm-up runs per scenario
const MIN_SECONDS_PER_PHASE: f64 = 0.75; // minimum wall-time per phase
const MIN_REPS_PER_PHASE: usize = 25; // minimum replicates per phase
const MAX_REPS_PER_PHASE: usize = 500; // safety cap on replicates

fn main() {
    if std::path::Path::new("benchmark_results.csv").exists() {
        println!(
            "Benchmark file already exists. Delete it if you want to run the benchmark again."
        );
        return;
    }
    let mut file = File::create("benchmark_results.csv").expect("Could not create benchmark file");
    let header = "Benchmarking Spatial Hash vs Naive Approach (post-warmup, time-budgeted runs)";
    println!("{}", header);
    // Write CSV header (expanded)
    writeln!(
        file,
        "phase,world_size,num_preds,num_preys,iteration,seed,cell_size,density_preds,density_preys,neighbors_pred_sh,neighbors_pred_naive,neighbors_prey_sh,neighbors_prey_naive,sh_ms,naive_ms,speedup"
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

    let world_sizes = vec![600.0, 1200.0, 2000.0, 4000.0];

    for world_size in world_sizes {
        // NOTE: These settings setters were used on a separate benchmarking branch where
        // screen size, sight range, and sight angle were changed to runtime-mutable (via RwLock).
        // The reason for this is that the setup should always benchmark runs on different devices.
        // On the main branch they are compile-time constants, so the setters no longer exist.
        if settings::SCREEN_WIDTH != world_size as i32 || settings::SCREEN_HEIGHT != world_size as i32 {
            println!("WARNING: World size settings do not match benchmark world size. Please ensure the code is up to date with the benchmarking branch.");
        }
        if settings::PRED_SIGHT_RANGE != 200.0 || settings::PREY_SIGHT_RANGE != 200.0 {
            println!("WARNING: Sight range settings do not match benchmark settings. Please ensure the code is up to date with the benchmarking branch.");
        }
        if settings::PRED_SIGHT_ANGLE != 60.0 || settings::PREY_SIGHT_ANGLE != 300.0 {
            println!("WARNING: Sight angle settings do not match benchmark settings. Please ensure the code is up to date with the benchmarking branch.");
        }
        if settings::PRED_SIGHT_ANGLE != 60.0 || settings::PREY_SIGHT_ANGLE != 300.0 {
            println!("WARNING: Sight angle settings do not match benchmark settings. Please ensure the code is up to date with the benchmarking branch.");
        }
        // settings::set_screen_width(world_size as i32);
        // settings::set_screen_height(world_size as i32);
        // settings::set_pred_sight_range(200.0);
        // settings::set_prey_sight_range(200.0);
        // settings::set_pred_sight_angle(60.0);
        // settings::set_prey_sight_angle(300.0);
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

    let mut total_sh_pred = Duration::ZERO;
    let mut total_naive_pred = Duration::ZERO;
    let mut total_sh_prey = Duration::ZERO;
    let mut total_naive_prey = Duration::ZERO;

    let mut neighbors_sh_pred_acc = 0usize;
    let mut neighbors_naive_pred_acc = 0usize;
    let mut neighbors_sh_prey_acc = 0usize;
    let mut neighbors_naive_prey_acc = 0usize;

    let mut reps_pred = 0usize;
    let mut reps_prey = 0usize;

    // Warm-ups: unrecorded runs to stabilize caches/branch prediction etc.
    for w in 0..WARMUP_RUNS {
        let mut rng = StdRng::seed_from_u64(42_000 + w as u64);

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

        let pred_sight_range = settings::PRED_SIGHT_RANGE;
        let cell_size_prey = (pred_sight_range.ceil() as usize).max(1);
        let mut hash_prey = SpatialHash::new(cell_size_prey, world_w, world_h);
        hash_prey.rebuild_from(&preys);
        let mut params_buffer = Vec::new();
        for pred in &predators {
            params_buffer.clear();
            hash_prey.query_into(&mut params_buffer, pred.core.x(), pred.core.y());
            for &idx in &params_buffer {
                let prey = &preys[idx];
                let _ = wrapped_distance_vector(pred.core.pos, prey.core.pos, world_w, world_h);
            }
        }

        let prey_sight_range = settings::PREY_SIGHT_RANGE;
        let cell_size_pred = (prey_sight_range.ceil() as usize).max(1);
        let mut hash_pred = SpatialHash::new(cell_size_pred, world_w, world_h);
        hash_pred.rebuild_from(&predators);
        for prey in &preys {
            params_buffer.clear();
            hash_pred.query_into(&mut params_buffer, prey.core.x(), prey.core.y());
            for &idx in &params_buffer {
                let pred = &predators[idx];
                let _ = wrapped_distance_vector(prey.core.pos, pred.core.pos, world_w, world_h);
            }
        }
    }

    // ----------------------------
    // Predator sensing Prey phase
    // ----------------------------
    let mut accum_time_sh_pred = Duration::ZERO;
    let mut accum_time_naive_pred = Duration::ZERO;
    while (reps_pred < MIN_REPS_PER_PHASE
        || accum_time_sh_pred.as_secs_f64() < MIN_SECONDS_PER_PHASE
        || accum_time_naive_pred.as_secs_f64() < MIN_SECONDS_PER_PHASE)
        && reps_pred < MAX_REPS_PER_PHASE
    {
        let seed = 42 + reps_pred as u64;
        let mut rng = StdRng::seed_from_u64(seed);

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

        let pred_sight_range = settings::PRED_SIGHT_RANGE;
        let pred_sight_angle = settings::PRED_SIGHT_ANGLE;
        let cell_size_prey = (pred_sight_range.ceil() as usize).max(1);
        let mut hash_prey = SpatialHash::new(cell_size_prey, world_w, world_h);

        let mut params_buffer = Vec::new();

        // SH build
        let start_sh_build_pred = Instant::now();
        hash_prey.rebuild_from(&preys);
        let elapsed_sh_build_pred = start_sh_build_pred.elapsed();

        // SH query
        let start_sh_query_pred = Instant::now();
        let mut neighbors_sh_pred_iter = 0usize;
        for pred in &predators {
            params_buffer.clear();
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
                        neighbors_sh_pred_iter += 1;
                    }
                }
            }
        }
        let elapsed_sh_query_pred = start_sh_query_pred.elapsed();
        let elapsed_sh_total_pred = elapsed_sh_build_pred + elapsed_sh_query_pred;

        // Naive query (predators sensing preys)
        let start_naive_pred = Instant::now();
        let mut neighbors_naive_pred_iter = 0usize;
        for pred in &predators {
            for prey in &preys {
                let delta = wrapped_distance_vector(pred.core.pos, prey.core.pos, world_w, world_h);
                let dist = delta.length();
                if dist <= pred_sight_range && dist > 0.0 {
                    let angle_to_prey = delta.y.atan2(delta.x);
                    let half_fov = (pred_sight_angle / 2.0).to_radians();
                    let diff = normalize_angle(angle_to_prey - pred.core.angle).abs();
                    if diff <= half_fov {
                        neighbors_naive_pred_iter += 1;
                    }
                }
            }
        }
        let elapsed_naive_pred = start_naive_pred.elapsed();

        // Densities
        let area = (world_size as f64) * (world_size as f64);
        let density_preds = (num_preds as f64) / area;
        let density_preys = (num_preys as f64) / area;

        // CSV rows for predator-sense phases
        let cell_size_prey_val = cell_size_prey as u32; // for consistent formatting
        // build
        writeln!(
            file,
            "pred_sense_build,{},{},{},{},{},{},{:.8},{:.8},{},{},{},{},{:.3},{:.3},{:.2}",
            world_size,
            num_preds,
            num_preys,
            reps_pred,
            seed,
            cell_size_prey_val,
            density_preds,
            density_preys,
            neighbors_sh_pred_iter,
            0,
            0,
            0,
            elapsed_sh_build_pred.as_secs_f64() * 1000.0,
            0.0,
            0.0,
        )
        .unwrap();
        // query
        writeln!(
            file,
            "pred_sense_query,{},{},{},{},{},{},{:.8},{:.8},{},{},{},{},{:.3},{:.3},{:.2}",
            world_size,
            num_preds,
            num_preys,
            reps_pred,
            seed,
            cell_size_prey_val,
            density_preds,
            density_preys,
            neighbors_sh_pred_iter,
            neighbors_naive_pred_iter,
            0,
            0,
            elapsed_sh_query_pred.as_secs_f64() * 1000.0,
            elapsed_naive_pred.as_secs_f64() * 1000.0,
            (elapsed_naive_pred.as_secs_f64()) / elapsed_sh_query_pred.as_secs_f64().max(1e-9),
        )
        .unwrap();
        // total
        writeln!(
            file,
            "pred_sense_total,{},{},{},{},{},{},{:.8},{:.8},{},{},{},{},{:.3},{:.3},{:.2}",
            world_size,
            num_preds,
            num_preys,
            reps_pred,
            seed,
            cell_size_prey_val,
            density_preds,
            density_preys,
            neighbors_sh_pred_iter,
            neighbors_naive_pred_iter,
            0,
            0,
            elapsed_sh_total_pred.as_secs_f64() * 1000.0,
            elapsed_naive_pred.as_secs_f64() * 1000.0,
            (elapsed_naive_pred.as_secs_f64()) / elapsed_sh_total_pred.as_secs_f64().max(1e-9),
        )
        .unwrap();

        total_sh_pred += elapsed_sh_total_pred;
        total_naive_pred += elapsed_naive_pred;
        accum_time_sh_pred += elapsed_sh_total_pred;
        accum_time_naive_pred += elapsed_naive_pred;
        neighbors_sh_pred_acc += neighbors_sh_pred_iter;
        neighbors_naive_pred_acc += neighbors_naive_pred_iter;
        reps_pred += 1;
    }

    // ----------------------------
    // Prey sensing Predators phase
    // ----------------------------
    let mut accum_time_sh_prey = Duration::ZERO;
    let mut accum_time_naive_prey = Duration::ZERO;
    while (reps_prey < MIN_REPS_PER_PHASE
        || accum_time_sh_prey.as_secs_f64() < MIN_SECONDS_PER_PHASE
        || accum_time_naive_prey.as_secs_f64() < MIN_SECONDS_PER_PHASE)
        && reps_prey < MAX_REPS_PER_PHASE
    {
        let seed = 84 + reps_prey as u64;
        let mut rng = StdRng::seed_from_u64(seed);

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

        let mut params_buffer = Vec::new();

        let prey_sight_range = settings::PREY_SIGHT_RANGE;
        let prey_sight_angle = settings::PREY_SIGHT_ANGLE;
        let cell_size_pred = (prey_sight_range.ceil() as usize).max(1);
        let mut hash_pred = SpatialHash::new(cell_size_pred, world_w, world_h);

        // SH build
        let start_sh_build_prey = Instant::now();
        hash_pred.rebuild_from(&predators);
        let elapsed_sh_build_prey = start_sh_build_prey.elapsed();

        // SH query
        let start_sh_query_prey = Instant::now();
        let mut neighbors_sh_prey_iter = 0usize;
        for prey in &preys {
            params_buffer.clear();
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
                        neighbors_sh_prey_iter += 1;
                    }
                }
            }
        }
        let elapsed_sh_query_prey = start_sh_query_prey.elapsed();
        let elapsed_sh_total_prey = elapsed_sh_build_prey + elapsed_sh_query_prey;

        // Naive query (preys sensing predators)
        let start_naive_prey = Instant::now();
        let mut neighbors_naive_prey_iter = 0usize;
        for prey in &preys {
            for pred in &predators {
                let delta = wrapped_distance_vector(prey.core.pos, pred.core.pos, world_w, world_h);
                let dist = delta.length();
                if dist <= prey_sight_range && dist > 0.0 {
                    let angle_to_pred = delta.y.atan2(delta.x);
                    let half_fov = (prey_sight_angle / 2.0).to_radians();
                    let diff = normalize_angle(angle_to_pred - prey.core.angle).abs();
                    if diff <= half_fov {
                        neighbors_naive_prey_iter += 1;
                    }
                }
            }
        }
        let elapsed_naive_prey = start_naive_prey.elapsed();

        // Densities
        let area = (world_size as f64) * (world_size as f64);
        let density_preds = (num_preds as f64) / area;
        let density_preys = (num_preys as f64) / area;

        // CSV rows for prey-sense phases
        let cell_size_pred_val = cell_size_pred as u32;
        // build
        writeln!(
            file,
            "prey_sense_build,{},{},{},{},{},{},{:.8},{:.8},{},{},{},{},{:.3},{:.3},{:.2}",
            world_size,
            num_preds,
            num_preys,
            reps_prey,
            seed,
            cell_size_pred_val,
            density_preds,
            density_preys,
            0,
            0,
            neighbors_sh_prey_iter,
            0,
            elapsed_sh_build_prey.as_secs_f64() * 1000.0,
            0.0,
            0.0,
        )
        .unwrap();
        // query
        writeln!(
            file,
            "prey_sense_query,{},{},{},{},{},{},{:.8},{:.8},{},{},{},{},{:.3},{:.3},{:.2}",
            world_size,
            num_preds,
            num_preys,
            reps_prey,
            seed,
            cell_size_pred_val,
            density_preds,
            density_preys,
            0,
            0,
            neighbors_sh_prey_iter,
            neighbors_naive_prey_iter,
            elapsed_sh_query_prey.as_secs_f64() * 1000.0,
            elapsed_naive_prey.as_secs_f64() * 1000.0,
            (elapsed_naive_prey.as_secs_f64()) / elapsed_sh_query_prey.as_secs_f64().max(1e-9),
        )
        .unwrap();
        // total
        writeln!(
            file,
            "prey_sense_total,{},{},{},{},{},{},{:.8},{:.8},{},{},{},{},{:.3},{:.3},{:.2}",
            world_size,
            num_preds,
            num_preys,
            reps_prey,
            seed,
            cell_size_pred_val,
            density_preds,
            density_preys,
            0,
            0,
            neighbors_sh_prey_iter,
            neighbors_naive_prey_iter,
            elapsed_sh_total_prey.as_secs_f64() * 1000.0,
            elapsed_naive_prey.as_secs_f64() * 1000.0,
            (elapsed_naive_prey.as_secs_f64()) / elapsed_sh_total_prey.as_secs_f64().max(1e-9),
        )
        .unwrap();

        total_sh_prey += elapsed_sh_total_prey;
        total_naive_prey += elapsed_naive_prey;
        accum_time_sh_prey += elapsed_sh_total_prey;
        accum_time_naive_prey += elapsed_naive_prey;
        neighbors_sh_prey_acc += neighbors_sh_prey_iter;
        neighbors_naive_prey_acc += neighbors_naive_prey_iter;
        reps_prey += 1;
    }

    // Summary statistics
    let avg_sh_pred = if reps_pred > 0 { total_sh_pred / reps_pred as u32 } else { Duration::ZERO };
    let avg_naive_pred = if reps_pred > 0 { total_naive_pred / reps_pred as u32 } else { Duration::ZERO };
    let avg_sh_prey = if reps_prey > 0 { total_sh_prey / reps_prey as u32 } else { Duration::ZERO };
    let avg_naive_prey = if reps_prey > 0 { total_naive_prey / reps_prey as u32 } else { Duration::ZERO };

    println!("  [Predator sensing Prey]");
    println!(
        "    -> Neighbors found (sum over iterations): SH={} Naive={}",
        neighbors_sh_pred_acc, neighbors_naive_pred_acc
    );
    println!(
        "    -> Reps: {} (min {} / time {:.2}s)",
        reps_pred,
        MIN_REPS_PER_PHASE,
        accum_time_sh_pred.as_secs_f64()
    );
    println!("    -> Avg. SH Time: {:.3?}", avg_sh_pred);
    println!("    -> Avg. Naive Time: {:.3?}", avg_naive_pred);
    println!(
        "    -> Avg. Speedup: {:.2}x",
        avg_naive_pred.as_secs_f64() / avg_sh_pred.as_secs_f64().max(1e-9)
    );

    println!("  [Prey sensing Predator]");
    println!(
        "    -> Neighbors found (sum over iterations): SH={} Naive={}",
        neighbors_sh_prey_acc, neighbors_naive_prey_acc
    );
    println!(
        "    -> Reps: {} (min {} / time {:.2}s)",
        reps_prey,
        MIN_REPS_PER_PHASE,
        accum_time_sh_prey.as_secs_f64()
    );
    println!("    -> Avg. SH Time: {:.3?}", avg_sh_prey);
    println!("    -> Avg. Naive Time: {:.3?}", avg_naive_prey);
    println!(
        "    -> Avg. Speedup: {:.2}x",
        avg_naive_prey.as_secs_f64() / avg_sh_prey.as_secs_f64().max(1e-9)
    );
}
