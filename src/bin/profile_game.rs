use predator_vs_prey::game::Game;
use predator_vs_prey::settings;
use std::time::Instant;

fn main() {
    let num_preds = 200; // Increase load for profiling
    let num_preys = 400;

    // Create a game with deterministic settings if possible, or just default.
    // We'll use the constructor that allows specifying counts.
    let mut game = Game::new(
        None,
        num_preds,
        num_preys,
        settings::MAX_PRED_COUNT,
        settings::MAX_PREY_COUNT,
        settings::SEED,
    );

    println!("Starting profiling run...");
    println!(
        "Initial State: {} Predators, {} Preys",
        game.predator_count(),
        game.prey_count()
    );

    let total_frames = 2000;
    let start_time = Instant::now();

    for i in 0..total_frames {
        game.next_frame_sequential();

        if i % 100 == 0 {
            println!(
                "Frame {}: {} preds, {} preys",
                i,
                game.predator_count(),
                game.prey_count()
            );
        }
    }

    let duration = start_time.elapsed();
    let total_secs = duration.as_secs_f64();
    let fps = total_frames as f64 / total_secs;

    println!("--------------------------------------------------");
    println!("Profiling Complete");
    println!("Total Time: {:.4}s", total_secs);
    println!("Total Frames: {}", total_frames);
    println!("Average FPS: {:.2}", fps);
    println!(
        "Final State: {} Predators, {} Preys",
        game.predator_count(),
        game.prey_count()
    );
    println!("--------------------------------------------------");

    println!("\nProfiling Breakdown:");
    coarse_prof::write(&mut std::io::stdout()).unwrap();
}
