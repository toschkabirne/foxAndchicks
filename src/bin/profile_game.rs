use predator_vs_prey::{game::Game, settings};
use std::hint::black_box;
use std::time::Instant;

const NUM_PREDS: usize = 200;
const NUM_PREYS: usize = 400;

const WARMUP_FRAMES: usize = 200;
const MEASURE_FRAMES: usize = 2000;
const RUNS: usize = 7;

// Wenn true: nach JEDEM Run coarse_prof Breakdown drucken (viel Output).
// Wenn false: nur für einen "repräsentativen" Run (z.B. Run 1).
const PRINT_BREAKDOWN_EACH_RUN: bool = true;

fn build_game() -> Game {
    Game::new(
        None,
        NUM_PREDS,
        NUM_PREYS,
        settings::MAX_PRED_COUNT,
        settings::MAX_PREY_COUNT,
        settings::SEED,
    )
}

fn step(game: &mut Game, frames: usize) {
    for _ in 0..frames {
        let frame = game.next_frame_sequential(); // nutzt deine coarse_prof::profile! scopes
        black_box(frame);
    }
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

fn main() {
    let mut secs: Vec<f64> = Vec::with_capacity(RUNS);
    let mut fps: Vec<f64> = Vec::with_capacity(RUNS);

    println!(
        "Profiling: {} preds, {} preys | warmup {} | measure {} | runs {}",
        NUM_PREDS, NUM_PREYS, WARMUP_FRAMES, MEASURE_FRAMES, RUNS
    );

    for run in 0..RUNS {
        let mut game = build_game();

        // Warmup (Caches/CPU hochfahren, JIT gibt’s nicht, aber Branch predictor & caches existieren)
        step(&mut game, WARMUP_FRAMES);

        // WICHTIG: Warmup aus dem Profiling rauswerfen
        coarse_prof::reset(); // Reset profiling information :contentReference[oaicite:1]{index=1}

        let start = Instant::now();
        step(&mut game, MEASURE_FRAMES);
        let dt = start.elapsed().as_secs_f64();

        let frames_per_sec = MEASURE_FRAMES as f64 / dt;
        secs.push(dt);
        fps.push(frames_per_sec);

        println!(
            "\n--------------------------------------------------\n\
             Run {} Complete\n\
             Total Time: {:.4}s\n\
             Total Frames: {}\n\
             Average FPS: {:.2}\n\
             Final State: {} Predators, {} Preys\n\
             --------------------------------------------------",
            run + 1,
            dt,
            MEASURE_FRAMES,
            frames_per_sec,
            game.predator_count(),
            game.prey_count()
        );

        if PRINT_BREAKDOWN_EACH_RUN || run == 0 {
            println!("\nProfiling Breakdown (measured window only):");
            coarse_prof::write(&mut std::io::stdout()).unwrap(); // coarse-prof hierarchical timing :contentReference[oaicite:2]{index=2}
        }

        // Damit der nächste Run nicht auf altem Profiling-Müll aufbaut
        coarse_prof::reset();
    }

    let (mean_s, std_s) = mean_std(&secs);
    let (mean_f, std_f) = mean_std(&fps);

    let min_f = fps.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_f = fps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!(
        "\n==================== Summary ====================\n\
         Runs: {}\n\
         Time:   mean {:.4}s  (std {:.4}s)\n\
         Speed:  mean {:.2} FPS (std {:.2}) | min {:.2}, max {:.2}\n\
         =================================================\n",
        RUNS, mean_s, std_s, mean_f, std_f, min_f, max_f
    );
}
