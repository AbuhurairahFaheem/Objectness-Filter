import cv2
import numpy as np
from .base_cue import BaseCue
from utils.helpers import IntegralImage

class MultiScaleSaliency(BaseCue):
    def precompute(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Process at a fixed scale for frequency analysis
        resized = cv2.resize(gray, (64, 64))
        
        fft = np.fft.fft2(resized)
        log_amplitude = np.log(np.abs(fft) + 1e-9)
        phase = np.angle(fft)
        
        # Spectral Residual = Amplitude - Average Filtered Amplitude
        avg_amplitude = cv2.blur(log_amplitude, (3, 3))
        residual = log_amplitude - avg_amplitude
        
        # Reconstruct saliency map
        saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase)))
        saliency = cv2.GaussianBlur(saliency**2, (3, 3), 0)
        
        full_saliency = cv2.resize(saliency, (self.width, self.height))
        self.ii = IntegralImage(full_saliency)

    def score(self, window):
        y1, x1, y2, x2 = window
        area = (y2 - y1) * (x2 - x1)
        return self.ii.get_sum(y1, x1, y2, x2) / (area + 1e-9)