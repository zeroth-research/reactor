"""Reusable gym.Env fixtures for the test suite.

We keep this minimal: most analyzer-feature tests define their own tiny env
inline (in test_analyzer.py). SimpleEnv is the one we share — used by
test_env.py (Env-class surface) and test_training.py (DQN training).

Both are defined beside the modules that wrap them (`fixtures/simple_env.py`,
`fixtures/twobit_env.py`), which import nothing from this package: the
`lean` tests load those as files, where `tests` is not this package."""

from tests.fixtures.simple_env import SimpleEnv
from tests.fixtures.twobit_env import TwoBitCounterEnv

__all__ = ["SimpleEnv", "TwoBitCounterEnv"]
