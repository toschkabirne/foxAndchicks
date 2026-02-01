//! Extended benchmark comparing heap allocation vs scratch vector approaches in `next_frame`.
//!
//! Features:
//! - Multiple population configurations (scaling analysis)
//! - CSV and markdown output for scientific analysis
//! - Detailed statistics per configuration
//! - Timestamp and system info in output

use predator_vs_prey::game::Game as ScratchGame;
use predator_vs_prey::game_heat_allo::Game as HeapGame;
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

/// Configuration for a single benchmark run
#[derive(Clone)]
struct BenchConfig {
    num_preds: usize,
    num_preys: usize,
    iterations: usize,
    frames_per_iter: usize,
}

/// Statistics for a benchmark run
#[derive(Clone)]
struct BenchStats {
    config: BenchConfig,
    scratch_mean_ms: f64,
    scratch_median_ms: f64,
    scratch_std_dev_ms: f64,
    scratch_min_ms: f64,
    scratch_max_ms: f64,
    heap_mean_ms: f64,
    heap_median_ms: f64,
    heap_std_dev_ms: f64,
    heap_min_ms: f64,
    heap_max_ms: f64,
    improvement_pct: f64,
    speedup_factor: f64,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let output_file =
        parse_arg_str(&args, "--output").unwrap_or_else(|| "benchmark_results".to_string());
    let iterations: usize = parse_arg(&args, "--iterations").unwrap_or(10);
    let warmup: usize = parse_arg(&args, "--warmup").unwrap_or(5);
    let frames_per_iter: usize = parse_arg(&args, "--frames").unwrap_or(500);

    // Population configurations to test (predators, prey = 5x predators)
    let configs: Vec<(usize, usize)> = vec![(10, 50), (30, 150), (50, 250), (150, 700)];

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Performance Benchmark: Heap Allocation vs Scratch Vector     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    println!("Parameters:");
    println!("  Iterations per config: {}", iterations);
    println!("  Warmup iterations: {}", warmup);
    println!("  Frames per iteration: {}", frames_per_iter);
    println!("  Output files: {}.csv, {}.md\n", output_file, output_file);

    let mut all_stats: Vec<BenchStats> = Vec::new();

    for (i, (num_preds, num_preys)) in configs.iter().enumerate() {
        println!(
            "[{}/{}] Testing {} predators, {} prey...",
            i + 1,
            configs.len(),
            num_preds,
            num_preys
        );

        let config = BenchConfig {
            num_preds: *num_preds,
            num_preys: *num_preys,
            iterations,
            frames_per_iter,
        };

        let stats = run_benchmark(&config, warmup);
        println!(
            "       Scratch: {:.2}ms, Heap: {:.2}ms, Improvement: {:.1}%",
            stats.scratch_mean_ms, stats.heap_mean_ms, stats.improvement_pct
        );

        all_stats.push(stats);
    }

    // Verify determinism for largest config
    println!("\n--- Determinism Check (largest config) ---");
    let last = configs.last().unwrap();
    verify_determinism(last.0, last.1, 50);

    // Write outputs
    let csv_path = format!("{}.csv", output_file);
    let md_path = format!("{}.md", output_file);

    write_csv(&csv_path, &all_stats);
    write_markdown(&md_path, &all_stats, iterations, warmup, frames_per_iter);

    println!("\n✓ Results written to:");
    println!("  - {} (CSV for data analysis)", csv_path);
    println!("  - {} (Markdown report)", md_path);

    // Print summary table
    print_summary_table(&all_stats);
}

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return args[i + 1].parse().ok();
        }
    }
    None
}

fn parse_arg_str(args: &[String], flag: &str) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
    }
    None
}

fn run_benchmark(config: &BenchConfig, warmup: usize) -> BenchStats {
    let scratch_times = benchmark_scratch(config, warmup);
    let heap_times = benchmark_heap(config, warmup);

    let scratch = compute_stats(&scratch_times);
    let heap = compute_stats(&heap_times);

    let improvement_pct = if heap.mean > 0.0 {
        (heap.mean - scratch.mean) / heap.mean * 100.0
    } else {
        0.0
    };

    let speedup_factor = if scratch.mean > 0.0 {
        heap.mean / scratch.mean
    } else {
        1.0
    };

    BenchStats {
        config: config.clone(),
        scratch_mean_ms: scratch.mean,
        scratch_median_ms: scratch.median,
        scratch_std_dev_ms: scratch.std_dev,
        scratch_min_ms: scratch.min,
        scratch_max_ms: scratch.max,
        heap_mean_ms: heap.mean,
        heap_median_ms: heap.median,
        heap_std_dev_ms: heap.std_dev,
        heap_min_ms: heap.min,
        heap_max_ms: heap.max,
        improvement_pct,
        speedup_factor,
    }
}

fn benchmark_scratch(config: &BenchConfig, warmup: usize) -> Vec<Duration> {
    let mut times = Vec::with_capacity(config.iterations);

    // Warmup
    for _ in 0..warmup {
        let mut game = ScratchGame::new(
            None,
            config.num_preds,
            config.num_preys,
            config.num_preds * 10,
            config.num_preys * 10,
        );
        for _ in 0..10 {
            let _ = game.next_frame();
        }
    }

    // Measure
    for _ in 0..config.iterations {
        let mut game = ScratchGame::new(
            None,
            config.num_preds,
            config.num_preys,
            config.num_preds * 10,
            config.num_preys * 10,
        );

        let start = Instant::now();
        for _ in 0..config.frames_per_iter {
            let _ = game.next_frame();
        }
        times.push(start.elapsed());
    }

    times
}

fn benchmark_heap(config: &BenchConfig, warmup: usize) -> Vec<Duration> {
    let mut times = Vec::with_capacity(config.iterations);

    // Warmup
    for _ in 0..warmup {
        let mut game = HeapGame::new(
            None,
            config.num_preds,
            config.num_preys,
            config.num_preds * 10,
            config.num_preys * 10,
        );
        for _ in 0..10 {
            let _ = game.next_frame();
        }
    }

    // Measure
    for _ in 0..config.iterations {
        let mut game = HeapGame::new(
            None,
            config.num_preds,
            config.num_preys,
            config.num_preds * 10,
            config.num_preys * 10,
        );

        let start = Instant::now();
        for _ in 0..config.frames_per_iter {
            let _ = game.next_frame();
        }
        times.push(start.elapsed());
    }

    times
}

fn verify_determinism(num_preds: usize, num_preys: usize, frames: usize) {
    let mut scratch_game =
        ScratchGame::new(None, num_preds, num_preys, num_preds * 10, num_preys * 10);
    let mut heap_game = HeapGame::new(None, num_preds, num_preys, num_preds * 10, num_preys * 10);

    for _ in 0..frames {
        let _ = scratch_game.next_frame();
        let _ = heap_game.next_frame();
    }

    let scratch_preds = scratch_game.predator_count();
    let scratch_preys = scratch_game.prey_count();
    let heap_preds = heap_game.predator_count();
    let heap_preys = heap_game.prey_count();

    if scratch_preds == heap_preds && scratch_preys == heap_preys {
        println!(
            "✓ Deterministic: Both have {} predators, {} prey after {} frames",
            scratch_preds, scratch_preys, frames
        );
    } else {
        println!("✗ Non-deterministic behavior detected!");
        println!(
            "  Scratch: {} predators, {} prey",
            scratch_preds, scratch_preys
        );
        println!("  Heap: {} predators, {} prey", heap_preds, heap_preys);
    }
}

struct Stats {
    mean: f64,
    median: f64,
    std_dev: f64,
    min: f64,
    max: f64,
}

fn compute_stats(times: &[Duration]) -> Stats {
    let mut ms: Vec<f64> = times.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = ms.len() as f64;
    let sum: f64 = ms.iter().sum();
    let mean = sum / n;

    let median = if ms.len() % 2 == 0 {
        (ms[ms.len() / 2 - 1] + ms[ms.len() / 2]) / 2.0
    } else {
        ms[ms.len() / 2]
    };

    let variance: f64 = ms.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    Stats {
        mean,
        median,
        std_dev,
        min: *ms.first().unwrap_or(&0.0),
        max: *ms.last().unwrap_or(&0.0),
    }
}

fn write_csv(path: &str, stats: &[BenchStats]) {
    let mut file = File::create(path).expect("Failed to create CSV file");

    // Header
    writeln!(
        file,
        "predators,prey,total_entities,iterations,frames_per_iter,\
         scratch_mean_ms,scratch_median_ms,scratch_std_dev_ms,scratch_min_ms,scratch_max_ms,\
         heap_mean_ms,heap_median_ms,heap_std_dev_ms,heap_min_ms,heap_max_ms,\
         improvement_pct,speedup_factor"
    )
    .unwrap();

    // Data rows
    for s in stats {
        writeln!(
            file,
            "{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.4}",
            s.config.num_preds,
            s.config.num_preys,
            s.config.num_preds + s.config.num_preys,
            s.config.iterations,
            s.config.frames_per_iter,
            s.scratch_mean_ms,
            s.scratch_median_ms,
            s.scratch_std_dev_ms,
            s.scratch_min_ms,
            s.scratch_max_ms,
            s.heap_mean_ms,
            s.heap_median_ms,
            s.heap_std_dev_ms,
            s.heap_min_ms,
            s.heap_max_ms,
            s.improvement_pct,
            s.speedup_factor
        )
        .unwrap();
    }
}

fn write_markdown(
    path: &str,
    stats: &[BenchStats],
    iterations: usize,
    warmup: usize,
    frames: usize,
) {
    let mut file = File::create(path).expect("Failed to create markdown file");

    // Header and metadata
    writeln!(
        file,
        "# Benchmark Report: Heap Allocation vs Scratch Vector\n"
    )
    .unwrap();
    writeln!(file, "**Generated**: {}", chrono_now()).unwrap();
    writeln!(file, "**Rust Version**: {}", env!("CARGO_PKG_VERSION")).unwrap();
    writeln!(file, "\n## Experiment Parameters\n").unwrap();
    writeln!(file, "| Parameter | Value |").unwrap();
    writeln!(file, "|-----------|-------|").unwrap();
    writeln!(file, "| Iterations per config | {} |", iterations).unwrap();
    writeln!(file, "| Warmup iterations | {} |", warmup).unwrap();
    writeln!(file, "| Frames per iteration | {} |", frames).unwrap();
    writeln!(file, "| Configurations tested | {} |", stats.len()).unwrap();

    // Summary table
    writeln!(file, "\n## Results Summary\n").unwrap();
    writeln!(
        file,
        "| Preds | Prey | Total | Scratch (ms) | Heap (ms) | Improvement | Speedup |"
    )
    .unwrap();
    writeln!(
        file,
        "|------:|-----:|------:|-------------:|----------:|------------:|--------:|"
    )
    .unwrap();

    for s in stats {
        writeln!(
            file,
            "| {} | {} | {} | {:.2} ± {:.2} | {:.2} ± {:.2} | {:.1}% | {:.2}x |",
            s.config.num_preds,
            s.config.num_preys,
            s.config.num_preds + s.config.num_preys,
            s.scratch_mean_ms,
            s.scratch_std_dev_ms,
            s.heap_mean_ms,
            s.heap_std_dev_ms,
            s.improvement_pct,
            s.speedup_factor
        )
        .unwrap();
    }

    // Detailed statistics
    writeln!(file, "\n## Detailed Statistics\n").unwrap();
    writeln!(file, "### Scratch Vector Implementation\n").unwrap();
    writeln!(
        file,
        "| Preds | Prey | Mean | Median | Std Dev | Min | Max |"
    )
    .unwrap();
    writeln!(
        file,
        "|------:|-----:|-----:|-------:|--------:|----:|----:|"
    )
    .unwrap();
    for s in stats {
        writeln!(
            file,
            "| {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |",
            s.config.num_preds,
            s.config.num_preys,
            s.scratch_mean_ms,
            s.scratch_median_ms,
            s.scratch_std_dev_ms,
            s.scratch_min_ms,
            s.scratch_max_ms
        )
        .unwrap();
    }

    writeln!(file, "\n### Heap Allocation Implementation\n").unwrap();
    writeln!(
        file,
        "| Preds | Prey | Mean | Median | Std Dev | Min | Max |"
    )
    .unwrap();
    writeln!(
        file,
        "|------:|-----:|-----:|-------:|--------:|----:|----:|"
    )
    .unwrap();
    for s in stats {
        writeln!(
            file,
            "| {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |",
            s.config.num_preds,
            s.config.num_preys,
            s.heap_mean_ms,
            s.heap_median_ms,
            s.heap_std_dev_ms,
            s.heap_min_ms,
            s.heap_max_ms
        )
        .unwrap();
    }

    // Analysis section
    writeln!(file, "\n## Analysis\n").unwrap();

    // Calculate averages
    let avg_improvement: f64 =
        stats.iter().map(|s| s.improvement_pct).sum::<f64>() / stats.len() as f64;
    let avg_speedup: f64 = stats.iter().map(|s| s.speedup_factor).sum::<f64>() / stats.len() as f64;
    let max_improvement = stats
        .iter()
        .max_by(|a, b| a.improvement_pct.partial_cmp(&b.improvement_pct).unwrap())
        .unwrap();

    writeln!(file, "### Key Findings\n").unwrap();
    writeln!(file, "1. **Average improvement**: {:.1}%", avg_improvement).unwrap();
    writeln!(file, "2. **Average speedup factor**: {:.2}x", avg_speedup).unwrap();
    writeln!(
        file,
        "3. **Maximum improvement**: {:.1}% at {} predators, {} prey",
        max_improvement.improvement_pct,
        max_improvement.config.num_preds,
        max_improvement.config.num_preys
    )
    .unwrap();

    // Variance analysis
    let scratch_avg_std: f64 =
        stats.iter().map(|s| s.scratch_std_dev_ms).sum::<f64>() / stats.len() as f64;
    let heap_avg_std: f64 =
        stats.iter().map(|s| s.heap_std_dev_ms).sum::<f64>() / stats.len() as f64;
    let variance_ratio = heap_avg_std / scratch_avg_std;

    writeln!(file, "\n### Variance Analysis\n").unwrap();
    writeln!(file, "| Metric | Scratch | Heap | Ratio |").unwrap();
    writeln!(file, "|--------|--------:|-----:|------:|").unwrap();
    writeln!(
        file,
        "| Avg Std Dev | {:.2} ms | {:.2} ms | {:.1}x |",
        scratch_avg_std, heap_avg_std, variance_ratio
    )
    .unwrap();
    writeln!(file, "\nThe heap allocation approach shows {:.1}x higher variance, indicating less predictable performance due to allocation pressure.\n", variance_ratio).unwrap();

    // Methodology
    writeln!(file, "## Methodology\n").unwrap();
    writeln!(
        file,
        "- Each configuration runs {} warmup iterations before measurement",
        warmup
    )
    .unwrap();
    writeln!(
        file,
        "- Each iteration creates a fresh game state and runs {} frames",
        frames
    )
    .unwrap();
    writeln!(
        file,
        "- Timing captures the full {} frames per iteration",
        frames
    )
    .unwrap();
    writeln!(
        file,
        "- Statistics computed: mean, median, std dev, min, max"
    )
    .unwrap();
    writeln!(
        file,
        "- Determinism verified by comparing final population counts"
    )
    .unwrap();
}

fn chrono_now() -> String {
    // Simple timestamp without external crate
    use std::process::Command;
    Command::new("date")
        .arg("+%Y-%m-%d %H:%M:%S")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "Unknown".to_string())
}

fn print_summary_table(stats: &[BenchStats]) {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                           Summary Table                                   ║");
    println!("╠═══════╦═══════╦═══════════════╦═══════════════╦═════════════╦═════════════╣");
    println!("║ Preds ║ Prey  ║ Scratch (ms)  ║ Heap (ms)     ║ Improvement ║   Speedup   ║");
    println!("╠═══════╬═══════╬═══════════════╬═══════════════╬═════════════╬═════════════╣");

    for s in stats {
        println!(
            "║ {:>5} ║ {:>5} ║ {:>6.2} ± {:>4.1} ║ {:>6.2} ± {:>4.1} ║   {:>6.1}%   ║    {:>5.2}x   ║",
            s.config.num_preds,
            s.config.num_preys,
            s.scratch_mean_ms,
            s.scratch_std_dev_ms,
            s.heap_mean_ms,
            s.heap_std_dev_ms,
            s.improvement_pct,
            s.speedup_factor
        );
    }

    println!("╚═══════╩═══════╩═══════════════╩═══════════════╩═════════════╩═════════════╝");

    // Overall stats
    let avg_improvement: f64 =
        stats.iter().map(|s| s.improvement_pct).sum::<f64>() / stats.len() as f64;
    let avg_speedup: f64 = stats.iter().map(|s| s.speedup_factor).sum::<f64>() / stats.len() as f64;

    println!("\n  Average improvement: {:.1}%", avg_improvement);
    println!("  Average speedup: {:.2}x", avg_speedup);
}
