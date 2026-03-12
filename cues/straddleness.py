import numpy as np
from skimage.segmentation import felzenszwalb
from .base_cue import BaseCue
from utils.helpers import IntegralImage

class Straddleness(BaseCue):
    def precompute(self):
        # 1. Simplify segmentation to reduce label count
        # Larger 'scale' and 'min_size' create bigger blobs, forcing boxes to expand
        self.segments = felzenszwalb(self.image, scale=500, sigma=0.8, min_size=1000)
        self.labels = np.unique(self.segments)
        
        # 2. Pre-calculate Integral Images for each major superpixel
        self.ii_maps = {}
        self.total_areas = {}
        
        for l in self.labels:
            mask = (self.segments == l).astype(np.uint8)
            self.ii_maps[l] = IntegralImage(mask)
            self.total_areas[l] = np.sum(mask)

    def score(self, window):
        y1, x1, y2, x2 = window
        straddle_sum = 0
        
        # Only check labels that actually exist in the window area to save time
        roi_labels = np.unique(self.segments[y1:y2, x1:x2])
        
        for l in roi_labels:
            # O(1) lookup of how many pixels of this superpixel are inside the box
            n_in = self.ii_maps[l].get_sum(y1, x1, y2, x2)
            n_total = self.total_areas[l]
            
            # If the superpixel is partially in and partially out, it straddles
            if 0 < n_in < n_total:
                # Straddleness formula from the paper
                straddle_sum += (n_in / n_total)
        
        # Higher score = Less straddling (Better object fit)
        return 1 - (straddle_sum / (len(roi_labels) + 1e-9))