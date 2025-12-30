import os
import subprocess
import re
import matplotlib.pyplot as plt

# Configuration
# Recommended values from generation.py are 0.1, 0.3, 0.5, 1.0. We'll probe a range.
LAMBDA_WEIGHTS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
DATASET_REL_PATH = "c2ec/dev.txt"
DATASET_NAME = "c2ec_dev"
PROJECT_ROOT = "/home/dell/lfr/simple-csc"
RESULT_FILE_PATH = os.path.join(PROJECT_ROOT, f"results/Qwen2.5-7B/Qwen2.5-7B-v1/{DATASET_NAME}/prediction.result")

# Environment variables base
ENV = os.environ.copy()
ENV.update({
    "DISTORTION_MODE": "normalize",
    "DISTORTION_CLAMP_MIN": "-1.0",
    "FAITHFULNESS_CLAMP_MIN": "1.0",
    "FAITHFULNESS_CLAMP_MAX": "1.5",
})

def run_experiment(lambda_weight):
    print(f"\n========================================")
    print(f"Running experiment with LAMBDA_WEIGHT={lambda_weight}")
    print(f"========================================")
    
    ENV["LAMBDA_WEIGHT"] = str(lambda_weight)
    
    # Construct command for run.py
    # Replicating run_crf.sh logic
    cmd_run = [
        "python", "-u", "run.py",
        "--input-file", f"datasets/{DATASET_REL_PATH}",
        "--path", f"results/Qwen2.5-7B/Qwen2.5-7B-v1/{DATASET_NAME}",
        "--model-name", "../models/Qwen2.5-7B",
        "--prompted-model-name", "../models/Qwen2.5-7B",
        "--config-path", "configs/c2ec_config.yaml",
        "--n-observed-chars", "8",
        "--prefix-split", "\n",
        "--n-beam", "8",
        "--batch-size", "200",
        "--max-length", "256",
        "--max-sentences-per-batch", "24",
        "--alpha", "2.5",
        "--temperature", "1.5",
        "--use-faithfulness-reward", # It is true in run_crf.sh so we add the flag
        "--distortion-model-smoothing", "-15.0"
    ]
    
    print(f"Executing: {' '.join(cmd_run)}")
    subprocess.run(cmd_run, cwd=PROJECT_ROOT, env=ENV, check=True)
    
    # Construct command for evaluate.py
    # Logic from run_crf.sh for lemon_v2 (matches regex)
    cmd_eval = [
        "python", "eval/evaluate.py",
        "--gold", f"datasets/{DATASET_REL_PATH}",
        "--hypo", f"results/Qwen2.5-7B/Qwen2.5-7B-v1/{DATASET_NAME}/prediction.txt",
        "--to_halfwidth",
        "--ignore_unmatch_length",
        "--ignore_space"
    ]
    
    print(f"Executing: {' '.join(cmd_eval)}")
    subprocess.run(cmd_eval, cwd=PROJECT_ROOT, env=ENV, check=True)

def parse_results():
    if not os.path.exists(RESULT_FILE_PATH):
        print(f"Result file not found: {RESULT_FILE_PATH}")
        return None, None
        
    with open(RESULT_FILE_PATH, 'r') as f:
        content = f.read()
        
    # Regex to find metrics
    # char correction f1:	48.166
    char_f1_match = re.search(r"char correction f1:\s+([\d\.]+)", content)
    sent_f1_match = re.search(r"sentence correction f1:\s+([\d\.]+)", content)
    
    char_f1 = float(char_f1_match.group(1)) if char_f1_match else 0.0
    sent_f1 = float(sent_f1_match.group(1)) if sent_f1_match else 0.0
    
    return char_f1, sent_f1

def main():
    results_char = []
    results_sent = []
    
    print("Starting Lambda Weight Probe...")
    
    for lw in LAMBDA_WEIGHTS:
        try:
            run_experiment(lw)
            c_f1, s_f1 = parse_results()
            if c_f1 is None:
                print(f"Failed to parse results for lambda={lw}")
                continue
                
            results_char.append(c_f1)
            results_sent.append(s_f1)
            print(f"Result -> Lambda: {lw}, Char F1: {c_f1}, Sent F1: {s_f1}")
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment for lambda={lw}: {e}")
            # Append 0 or skip? Let's skip to keep lengths matched if we want to plot correctly
            # But better to handle gracefully. For now, just continue.
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue
        
    # Plotting
    if not results_char:
        print("No results collected.")
        return

    plt.figure(figsize=(10, 6))
    
    # Plot Char Correction F1
    plt.plot(LAMBDA_WEIGHTS[:len(results_char)], results_char, marker='^', linestyle='--', label='Char Correction F1')
    
    # Plot Sentence Correction F1
    plt.plot(LAMBDA_WEIGHTS[:len(results_sent)], results_sent, marker='o', linestyle='-', label='Sentence Correction F1')
    
    plt.xlabel('Lambda Weight')
    plt.ylabel('F1 Score')
    plt.title(f'Effect of Lambda Weight on (${DATASET_NAME})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_plot = os.path.join(PROJECT_ROOT, "lambda_weight_effect.png")
    plt.savefig(output_plot)
    print(f"\nPlot saved to {output_plot}")
    print("Done.")

if __name__ == "__main__":
    main()
