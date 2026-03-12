import argparse
import os
import cv2
import numpy as np
from cues.saliency import MultiScaleSaliency
from cues.edge_density import EdgeDensity
from cues.straddleness import Straddleness
from objectness import ObjectnessScorer
from utils.helpers import generate_windows

def nms(bboxes, scores, threshold=0.2):
    """Simple Non-Maximum Suppression to avoid overlapping boxes."""
    x1, y1, x2, y2 = bboxes[:, 1], bboxes[:, 0], bboxes[:, 3], bboxes[:, 2]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return keep

def process_image(img_path, output_folder):
    img = cv2.imread(img_path)
    if img is None: return
    
    # --- PERFORMANCE: Resize to 500px width ---
    # This is the secret to making Straddleness fast!
    scale = 500.0 / img.shape[1]
    img_small = cv2.resize(img, (500, int(img.shape[0] * scale)))
    
    # 1. Generate 2000 diversified window proposals
    windows = generate_windows(img_small.shape[:2], num_windows=2000)
    
    # 2. Initialize the modular cues
    cues = [
        MultiScaleSaliency(img_small),
        EdgeDensity(img_small),
        Straddleness(img_small)
    ]
    
    # 3. Compute Bayesian scores
    scorer = ObjectnessScorer(cues)
    scores = scorer.compute_scores(windows)
    
    # 4. Apply NMS and Get Top 5
    keep_indices = nms(windows, scores, threshold=0.2)
    top_indices = keep_indices[:5]
    top_windows = windows[top_indices]
    
    # 5. Visualization
    canvas = img_small.copy()
    # Draw every 20th proposal in faint blue (to show the search space)
    for w in windows[::20]:
        cv2.rectangle(canvas, (w[1], w[0]), (w[3], w[2]), (255, 0, 0), 1) 
    
    # Draw Top 5 in Bold Green
    for w in top_windows:
        cv2.rectangle(canvas, (w[1], w[0]), (w[3], w[2]), (0, 255, 0), 2)
        
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