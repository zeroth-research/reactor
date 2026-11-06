"""`certify`: `reactor` and `verith` together, as one command.

`cli` is its command line, `reactor`'s flags and the routes'; `pipeline`
runs `reactor`'s stages and the selected route in one process. It imports
both packages and neither imports it.
"""

from .pipeline import main

__all__ = ["main"]
