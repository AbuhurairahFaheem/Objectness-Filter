import argparse
import os
import cv2
import numpy as np
from cues.saliency import MultiScaleSaliency
from cues.edge_density import EdgeDensity
from cues.straddleness import Straddleness
from objectness import ObjectnessScorer
from utils.helpers import generate_windows

def process_image(img_path, output_folder):
    img = cv2.imread(img_path)
    if img is None: return
    
    # 1. Generate 2000 diversified window proposals
    windows = generate_windows(img.shape[:2], num_windows=2000)
    
    # 2. Initialize the modular cues
    cues = [
        MultiScaleSaliency(img),
        EdgeDensity(img),
        Straddleness(img)
    ]
    
    # 3. Compute Bayesian scores and rank
    scorer = ObjectnessScorer(cues)
    scores = scorer.compute_scores(windows)
    top_windows, top_scores = scorer.get_top_n(windows, scores, n=5)
    
    # 4. Visualization for your report
    # Draw all 2000 (faint) and top 5 (bold)
    canvas = img.copy()
    for w in windows:
        cv2.rectangle(canvas, (w[1], w[0]), (w[3], w[2]), (255, 0, 0), 1) # Blue: Proposals
    
    for w in top_windows:
        cv2.rectangle(canvas, (w[1], w[0]), (w[3], w[2]), (0, 255, 0), 3) # Green: Top 5
        
    file_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_folder, file_name), canvas)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for f in os.listdir(args.input_folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {f}...")
            process_image(os.path.join(args.input_folder, f), args.output_folder)

if __name__ == "__main__":
    main()