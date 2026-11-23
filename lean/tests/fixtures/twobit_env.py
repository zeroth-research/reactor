"""TwoBitCounterEnv gym wrapper fixture.

2-bit digital counter with enable action.
Property: b0 = False holds infinitely often (when enable is always on).
"""
import gymnasium as gym
from gymnasium import spaces

from reactor.gym import Env
from reactor import Module, Bool, Int


class TwoBitCounterEnv(gym.Env):
    """2-bit counter: the action enables a tick; b0 toggles and carries into b1.

    The gym twin of the `twobit_lia` fixture's hand-built module — same two
    bits, same `enable`, so the two encodings of one system can be compared."""

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)  # enable: off / on
        self.observation_space = spaces.Discrete(2)

    def _get_observation(self):
        return 1 if self.b0 and self.b1 else 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.b0 = False
        self.b1 = False
        return self._get_observation(), {}

    def step(self, q_values):
        enable = q_values.argmax().item() == 1
        carry = self.b0 and enable  # reads b0 before the toggle below
        self.b0 = (not self.b0) if enable else self.b0
        self.b1 = (not self.b1) if carry else self.b1
        reward = 1.0 if self.b0 and self.b1 else 0.0
        return self._get_observation(), reward, False, False, {}


def module() -> Module:
    # `b0`/`b1` are TwoBitCounterEnv's private state; their sorts must be explicit.
    return Env(TwoBitCounterEnv(), attrs=Bool([1, 1]), number=Int)
