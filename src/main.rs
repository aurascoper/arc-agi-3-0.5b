use serde::{Deserialize, Serialize};
use std::fs;
use std::env;
use std::cmp;

// Define the structure of an ARC Task JSON
#[derive(Debug, Deserialize, Serialize)]
struct ArcTask {
    train: Vec<Example>,
    test: Vec<Example>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Example {
    input: Vec<Vec<i32>>,
    output: Vec<Vec<i32>>,
}

/// A highly optimized structural accuracy calculator
/// This replaces the slow Python Hausdorff calculation for local i5 testing
fn calculate_pixel_accuracy(pred: &Vec<Vec<i32>>, true_grid: &Vec<Vec<i32>>) -> f64 {
    // 1. Check strict structural failure (dimensions don't match)
    let true_rows = true_grid.len();
    if true_rows == 0 { return 0.0; }
    let true_cols = true_grid[0].len();
    
    let pred_rows = pred.len();
    if pred_rows == 0 || pred_rows != true_rows { return 0.0; }
    let pred_cols = pred[0].len();
    if pred_cols != true_cols { return 0.0; }

    // 2. Calculate soft topological overlap (Partial Credit)
    let mut total_pixels = 0;
    let mut correct_pixels = 0;

    for r in 0..true_rows {
        for c in 0..true_cols {
            total_pixels += 1;
            if pred[r][c] == true_grid[r][c] {
                correct_pixels += 1;
            }
        }
    }

    if total_pixels == 0 { return 0.0; }
    
    let accuracy = correct_pixels as f64 / total_pixels as f64;
    (accuracy * 100.0).round() / 100.0
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: arc_eval <path_to_task.json>");
        std::process::exit(1);
    }

    let file_path = &args[1];
    
    // Read and parse the ARC JSON
    let data = fs::read_to_string(file_path).expect("Unable to read file");
    let task: ArcTask = serde_json::from_str(&data).expect("JSON was not well-formatted");

    // Grab the first test example's output to use as our "True" grid
    let true_output = &task.test[0].output;

    // For testing on the library computer, we simulate Qwen's output 
    // by injecting a slightly flawed version of the true output
    let mut mock_qwen_prediction = true_output.clone();
    if mock_qwen_prediction.len() > 0 && mock_qwen_prediction[0].len() > 0 {
         // Deliberately flip one pixel to test the partial credit math
         mock_qwen_prediction[0][0] = 9; 
    }

    // Run the high-speed Rust evaluation
    let score = calculate_pixel_accuracy(&mock_qwen_prediction, true_output);

    println!("--- ARC Rust Evaluator ---");
    println!("Target Task: {}", file_path);
    println!("Grid Dimensions: {}x{}", true_output.len(), if true_output.len() > 0 { true_output[0].len() } else { 0 });
    println!("Topological Score: {}", score);
    
    // Exit with code 0 if passing the 0.85 threshold, else 1
    if score >= 0.85 {
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}