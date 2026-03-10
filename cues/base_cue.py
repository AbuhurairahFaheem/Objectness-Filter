from abc import ABC, abstractmethod

class BaseCue(ABC):
    def __init__(self, image):
        self.image = image
        self.height, self.width = image.shape[:2]
        self.precompute()

    @abstractmethod
    def precompute(self):
        """Logic to prepare maps or integral images."""
        pass

    @abstractmethod
    def score(self, window):
        """Returns a raw score for a window [y1, x1, y2, x2]."""
        pass