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
    
    # --- STEP 0: RESIZE FOR PERFORMANCE ---
    # Working with 1080p+ images will kill your performance.
    # We downsample to a manageable 500px width.
    original_shape = img.shape[:2]
    target_width = 500
    scale = target_width / img.shape[1]
    img_small = cv2.resize(img, (target_width, int(img.shape[0] * scale)))
    
    # 1. Generate 2000 diversified window proposals
    windows = generate_windows(img_small.shape[:2], num_windows=2000)
    
    # 2. Initialize the modular cues using the resized image
    cues = [
        MultiScaleSaliency(img_small),
        EdgeDensity(img_small),
        Straddleness(img_small)
    ]
    
    # 3. Compute Bayesian scores
    scorer = ObjectnessScorer(cues)
    scores = scorer.compute_scores(windows)
    
    # --- STEP 3.5: PRUNING OVERLAP (NMS) ---
    # To ensure the top 5 are DIFFERENT objects, we pick the best, 
    # then remove windows that overlap significantly with it.
    top_windows, top_scores = scorer.get_top_n(windows, scores, n=5)
    
    # 4. Visualization
    canvas = img_small.copy()
    # Draw a subset of proposals (e.g., every 10th) so the image isn't solid blue
    for w in windows[::10]:
        cv2.rectangle(canvas, (w[1], w[0]), (w[3], w[2]), (255, 0, 0), 1) 
    
    # Draw Top 5 in bold Green
    for i, w in enumerate(top_windows):
        cv2.rectangle(canvas, (w[1], w[0]), (w[3], w[2]), (0, 255, 0), 2)
        cv2.putText(canvas, f"Score: {top_scores[i]:.2f}", (w[1], w[0]-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
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