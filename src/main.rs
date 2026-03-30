use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::env;

// --- DATA STRUCTURES ---

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

// --- THE BRAIN (OLLAMA CALL VIA UREQ) ---

fn ask_qwen(input_grid: &Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let prompt = format!(
        "Return ONLY a JSON array representing the transformed ARC grid for this input: {:?}", 
        input_grid
    );

    println!("Sending grid to Qwen-0.5b...");

    // ureq is blocking by default, no tokio or client builder needed!
    let res: serde_json::Value = ureq::post("http://localhost:11434/api/generate")
        .send_json(json!({
            "model": "qwen2.5-coder:0.5b",
            "prompt": prompt,
            "stream": false,
            "format": "json" 
        }))
        .expect("Failed to connect to Ollama. Is 'ollama serve' running?")
        .into_json()
        .expect("Failed to parse response as JSON");

    let response_text = res["response"].as_str().unwrap_or("[]");
    
    serde_json::from_str(response_text).unwrap_or_else(|_| {
        println!("Warning: Qwen returned invalid grid format.");
        vec![vec![0]]
    })
}

// --- THE JUDGE (SCORING) ---

fn calculate_pixel_accuracy(pred: &Vec<Vec<i32>>, true_grid: &Vec<Vec<i32>>) -> f64 {
    let true_rows = true_grid.len();
    if true_rows == 0 { return 0.0; }
    let true_cols = true_grid[0].len();
    
    if pred.len() != true_rows || pred[0].len() != true_cols {
        println!("Structural failure: Prediction dimensions do not match target.");
        return 0.0; 
    }

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

    (correct_pixels as f64 / total_pixels as f64).round()
}

// --- MAIN EXECUTION LOOP ---

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: agi <path_to_task.json>");
        std::process::exit(1);
    }

    let file_path = &args[1];
    let data = fs::read_to_string(file_path).expect("Unable to read file");
    let task: ArcTask = serde_json::from_str(&data).expect("JSON was not well-formatted");

    let test_input = &task.test[0].input;
    let true_output = &task.test[0].output;

    let prediction = ask_qwen(test_input);
    println!("Qwen Prediction: {:?}", prediction);

    let score = calculate_pixel_accuracy(&prediction, true_output);

    println!("--- ARC Swarm Result ---");
    println!("Task: {}", file_path);
    println!("Topological Accuracy: {:.2}%", score * 100.0);
    
    if score >= 0.85 {
        println!("STATUS: PASS");
        std::process::exit(0);
    } else {
        println!("STATUS: FAIL");
        std::process::exit(1);
    }
}