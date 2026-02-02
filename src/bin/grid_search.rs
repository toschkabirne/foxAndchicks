use predator_vs_prey::game::Game;
use predator_vs_prey::settings;
use std::fs::OpenOptions;
use std::io::Write;

fn main() {
    let mut results_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true) // Overwrite if exists
        .open("grid_search_results.txt")
        .expect("Failed to open output file");

    let max_frames = 60_000;

    // for i in [61,66] {
    for i in (56..76).step_by(1) {
        let current_seed = i;

        // Print less verbose progress to terminal
        print!("Seed {}: ", current_seed);
        std::io::stdout().flush().unwrap();

        let mut game = Game::new(
            None,
            settings::PRED_INIT_NUMB,
            settings::PREY_INIT_NUMB,
            settings::MAX_PRED_COUNT,
            settings::MAX_PREY_COUNT,
            current_seed,
        );

        let mut survived = true;
        let mut fail_reason = "";

        for frame_num in 1..=max_frames {
            game.next_frame();

            if game.predator_count() == 0 {
                survived = false;
                fail_reason = "Predators died out";
                break;
            }
            if game.prey_count() == 0 {
                survived = false;
                fail_reason = "Prey died out";
                break;
            }

            // Optional: Print a dot every 2000 frames to show it's alive
            if frame_num % 5000 == 0 {
                print!(".");
                std::io::stdout().flush().unwrap();
            }
        }

        if survived {
            println!(" SUCCESS!");
            writeln!(results_file, "Seed {}: SUCCESS", current_seed).unwrap();
        } else {
            println!(" FAILED ({})", fail_reason);
            // We only need to save successful seeds, but maybe logging failures is good for debugging?
            // The prompt said "If scucesfully reached, we save the seed number."
            // So we strictly interpret this as: only save success.
            // However, the prompt also said "Write the results to an output file", implying potentially all results or just the "results of the search".
            // I'll stick to just writing success to be clean, or I can write everything.
            // "If scucesfully reached, we save the seed number." strongly suggests filtering.
            // But "Write the results to an output file" could mean a log.
            // I will write "FAILED" too so the user knows checks were run.
            writeln!(
                results_file,
                "Seed {}: FAILED ({})",
                current_seed, fail_reason
            )
            .unwrap();
        }
    }

    println!("Grid search complete. Results saved to grid_search_results.txt");
}
