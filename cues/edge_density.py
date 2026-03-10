import cv2
from .base_cue import BaseCue
from utils.helpers import IntegralImage

class EdgeDensity(BaseCue):
    def precompute(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Canny identifies high-gradient areas (potential object boundaries)
        edges = cv2.Canny(gray, 100, 200)
        self.ii = IntegralImage(edges > 0)

    def score(self, window):
        y1, x1, y2, x2 = window
        edge_count = self.ii.get_sum(y1, x1, y2, x2)
        area = (y2 - y1) * (x2 - x1)
        return edge_count / (area + 1e-9)