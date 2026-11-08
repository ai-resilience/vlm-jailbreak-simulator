"""Original CIDER cosine-drop decision rule."""
import numpy as np


class Defender:
    def __init__(self, threshold=None):
        self.threshold = threshold

    def get_lowest_idx(self, cossims):
        """
        :cossims: a nD array of cosine similarity (1*n)
            cols: denoise times n
            row: 1 text
        return: a nD array of int (1*1)
            0 is clean, positive int is the index of lowest cosine similarity
        """
        # reshape the data
        cossims = np.array(cossims).reshape(1, -1)
        delta = cossims - cossims[0, 0]
        if np.min(delta) < self.threshold:
            return np.argmin(delta)
        else:
            return 0
