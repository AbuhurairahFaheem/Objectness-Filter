import numpy as np
from skimage.segmentation import felzenszwalb
from .base_cue import BaseCue

class Straddleness(BaseCue):
    def precompute(self):
        # Segment image into superpixels
        self.segments = felzenszwalb(self.image, scale=100, sigma=0.5, min_size=50)
        self.labels = np.unique(self.segments)
        # Cache total area for each superpixel
        self.sp_areas = {l: np.sum(self.segments == l) for l in self.labels}

    def score(self, window):
        y1, x1, y2, x2 = window
        win_roi = self.segments[y1:y2, x1:x2]
        straddle_sum = 0
        
        # Check superpixels found within the window
        present_labels = np.unique(win_roi)
        for l in present_labels:
            area_inside = np.sum(win_roi == l)
            total_area = self.sp_areas[l]
            
            # If a superpixel is partially inside, it's straddling
            if 0 < area_inside < total_area:
                straddle_sum += (area_inside / total_area)
        
        return 1 - (straddle_sum / (len(present_labels) + 1e-9))