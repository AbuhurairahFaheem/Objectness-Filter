import numpy as np
# Import these so isinstance() can recognize them
from cues.saliency import MultiScaleSaliency
from cues.edge_density import EdgeDensity
from cues.straddleness import Straddleness

class ObjectnessScorer:
    def __init__(self, cues):
        self.cues = cues

    def compute_scores(self, windows):
        final_scores = np.ones(len(windows))

        for cue in self.cues:
            raw_scores = np.array([cue.score(w) for w in windows])
            
            # Normalize to [0, 1]
            denom = (raw_scores.max() - raw_scores.min() + 1e-9)
            norm_scores = (raw_scores - raw_scores.min()) / denom
            
            # BIAS LOGIC: 
            # We want Straddleness to have the most "vote" on the box size.
            if isinstance(cue, Straddleness):
                # Squaring a normalized value (0-1) makes it stricter.
                # Only boxes with EXCELLENT straddleness stay high.
                final_scores *= (norm_scores ** 2.0) 
            elif isinstance(cue, EdgeDensity):
                # Square root (0.5) makes the cue "gentler" so it doesn't
                # punish large boxes just because their center is empty.
                final_scores *= (norm_scores ** 0.5)
            else:
                final_scores *= norm_scores

        return final_scores # CRITICAL: Don't forget to return this!

    def get_top_n(self, windows, scores, n=5):
        indices = np.argsort(scores)[-n:][::-1]
        return windows[indices], scores[indices]