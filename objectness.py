import numpy as np

class ObjectnessScorer:
    def __init__(self, cues):
        """
        cues: List of initialized cue objects (Saliency, EdgeDensity, Straddleness)
        """
        self.cues = cues

    def compute_scores(self, windows):
        # Start with a uniform prior (1.0)
        final_scores = np.ones(len(windows))

        for cue in self.cues:
            # Vectorized scoring for all 2000 windows
            raw_scores = np.array([cue.score(w) for w in windows])
            
            # Min-Max Normalization to convert raw metrics into a [0, 1] probability space
            if raw_scores.max() != raw_scores.min():
                norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
            else:
                norm_scores = raw_scores
                
            # Bayesian product: p(obj|c1, c2, ...) ∝ p(c1|obj) * p(c2|obj) ...
            final_scores *= (norm_scores + 1e-6) 

        return final_scores

    def get_top_n(self, windows, scores, n=5):
        """Returns the n highest scoring windows."""
        indices = np.argsort(scores)[-n:][::-1]
        return windows[indices], scores[indices]