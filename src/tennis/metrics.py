"""Separate complete-episode rolling scores from partial/truncated returns."""
from collections import deque
import numpy as np


class ScoreWindow:
    def __init__(self, width=100):
        self.width = width
        self.values = deque(maxlen=width)

    def add(self, agent_returns, complete):
        returns = np.asarray(agent_returns, np.float64)
        if returns.shape != (2,) or not np.isfinite(returns).all():
            raise ValueError('Expected two finite undiscounted agent returns')
        score = float(returns.max())
        if complete:
            self.values.append(score)
        else:
            # A gap must break a consecutive-episode qualification window.
            self.values.clear()
        mean = float(np.mean(self.values)) if len(self.values) == self.width else None
        return score, mean
