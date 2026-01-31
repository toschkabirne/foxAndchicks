use predator_vs_prey::animals::{wrapped_distance_abs, Predator, Prey};
use predator_vs_prey::settings::{self};
use predator_vs_prey::spatial_hash::SpatialHash;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

fn main() {
    let mut file = File::create("benchmark_results.csv").expect("Could not create benchmark file");
    let header = "Benchmarking Spatial Hash vs Naive Approach (Average of 10 runs)";
    println!("{}", header);
    // Write CSV header
    writeln!(
        file,
        "world_size,num_preds,num_preys,iteration,sh_ms,naive_ms,speedup"
    )
    .unwrap();
    println!("--------------------------------------------------");

    let scenarios = vec![
        (10, 20),
        (50, 100),
        (100, 200),
        (500, 1000),
        (1000, 2000),
        (2000, 4000),
        (4000, 8000),
    ];

    let world_sizes = vec![2000.0];

    for world_size in world_sizes {
        settings::set_screen_width(world_size as i32);
        settings::set_screen_height(world_size as i32);
        settings::set_pred_sight_range(200.0);
        settings::set_prey_sight_range(200.0);
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
    let mut total_sh = Duration::ZERO;
    let mut total_naive = Duration::ZERO;
    let mut total_neighbors_sh = 0;
    let mut total_neighbors_naive = 0;

    for i in 0..iterations {
        // Setup entities with changing seed for variety but determinism across runs
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
        let sight_range = settings::pred_sight_range();

        // ----------------------------------------------------------------
        // 1. Spatial Hash Approach
        // ----------------------------------------------------------------
        // Note: Cell size MUST be at least as large as sight_range for a 3x3 query to find all neighbors.
        let cell_size = (sight_range.ceil() as i32).max(1);
        let mut hash = SpatialHash::new(cell_size, world_w, world_h);

        let start_sh = Instant::now();

        // Step A: Rebuild
        hash.rebuild_from(&preys);

        // Step B: Query for each predator
        // Buffer reuse to match game implementation optimization
        let mut params_buffer = Vec::new();

        for pred in &predators {
            // Query returns indices of potential neighbors
            hash.query_into(&mut params_buffer, pred.core.x(), pred.core.y());

            for &idx in &params_buffer {
                let prey = &preys[idx];
                if wrapped_distance_abs(pred.core.pos, prey.core.pos, world_w, world_h)
                    <= sight_range
                {
                    if i == 0 {
                        total_neighbors_sh += 1;
                    }
                }
            }
        }

        let elapsed_sh = start_sh.elapsed();
        total_sh += elapsed_sh;

        // ----------------------------------------------------------------
        // 2. Naive Approach
        // ----------------------------------------------------------------
        let start_naive = Instant::now();

        for pred in &predators {
            for prey in &preys {
                if wrapped_distance_abs(pred.core.pos, prey.core.pos, world_w, world_h)
                    <= sight_range
                {
                    if i == 0 {
                        total_neighbors_naive += 1;
                    }
                }
            }
        }

        let elapsed_naive = start_naive.elapsed();
        total_naive += elapsed_naive;

        // Write CSV row for each iteration
        writeln!(
            file,
            "{},{},{},{},{:.3},{:.3},{:.2}",
            world_size,
            num_preds,
            num_preys,
            i,
            elapsed_sh.as_secs_f64() * 1000.0,
            elapsed_naive.as_secs_f64() * 1000.0,
            elapsed_naive.as_secs_f64() / elapsed_sh.as_secs_f64()
        )
        .unwrap();
    }

    let avg_sh = total_sh / iterations;
    let avg_naive = total_naive / iterations;

    // ----------------------------------------------------------------
    // Report
    // ----------------------------------------------------------------
    let neighbors_line = format!(
        "  -> Neighbors found (Run 0): SH={} vs Naive={} (Should be identical)",
        total_neighbors_sh, total_neighbors_naive
    );
    let sh_time_line = format!("  -> Avg. Spatial Hash Time: {:.3?}", avg_sh);
    let naive_time_line = format!("  -> Avg. Naive Approach Time: {:.3?}", avg_naive);

    println!("{}", neighbors_line);
    println!("{}", sh_time_line);
    println!("{}", naive_time_line);

    let speedup = avg_naive.as_secs_f64() / avg_sh.as_secs_f64();
    let speedup_line = format!("  => Avg. Speedup: {:.2}x", speedup);
    println!("{}", speedup_line);
}
