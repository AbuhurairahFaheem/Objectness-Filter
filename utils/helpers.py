import numpy as np

class IntegralImage:
    """Computes Summed Area Table for O(1) window queries."""
    def __init__(self, data):
        # Pad with zeros to handle top/left boundary cases
        self.padded_ii = np.pad(data.astype(np.float64).cumsum(axis=0).cumsum(axis=1), 
                                ((1, 0), (1, 0)), mode='constant')

    def get_sum(self, y1, x1, y2, x2):
        """Returns the sum of pixels in a rectangle in O(1)."""
        return (self.padded_ii[y2+1, x2+1] - self.padded_ii[y1, x2+1] - 
                self.padded_ii[y2+1, x1] + self.padded_ii[y1, x1])

def generate_windows(img_shape, num_windows=2000):
    H, W = img_shape
    windows = []
    # Force include very large scales to capture the whole car
    scales = [0.3, 0.5, 0.7, 0.9] 
    # Cars are horizontal: width is much larger than height (Ratio = H/W)
    ratios = [0.3, 0.4, 0.5, 0.6] 
    
    for _ in range(num_windows):
        s = np.random.choice(scales)
        r = np.random.choice(ratios)
        
        w = int(W * s)
        h = int(w * r)
        
        if h < H and w < W:
            # Random jitter for placement
            y1 = np.random.randint(0, H - h)
            x1 = np.random.randint(0, W - w)
            windows.append([y1, x1, y1 + h, x1 + w])
            
    return np.array(windows)