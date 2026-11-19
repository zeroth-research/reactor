"""SimpleEnv gym wrapper fixture.

Chain environment: state ∈ {0,1,2}, action moves left/right.
Property: state reaches 2 infinitely often.
"""
import gymnasium as gym
from gymnasium import spaces

from reactor.gym import Env
from reactor import Module, Real


class SimpleEnv(gym.Env):
    """Chain env: state moves left/right; reward 1 at goal (state == 2)."""

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)

    def _get_observation(self):
        return 1 if self.state == 2 else 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self._get_observation(), {}

    def step(self, q_values):
        action = q_values.argmax().item()
        if action == 1:
            self.state = min(self.state + 1, 2)
        else:
            self.state = max(self.state - 1, 0)
        reward = 1.0 if self.state == 2 else 0.0
        terminated = self._get_observation() == 1
        return self._get_observation(), reward, terminated, False, {}


def module() -> Module:
    # `state` is SimpleEnv's only private attribute; its sort must be explicit.
    return Env(SimpleEnv(), attrs=Real([1, 1]))
