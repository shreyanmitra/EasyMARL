"""EasyMARL Environments"""

from .vectorized_env import *

try:
    from .gym_multigrid import *
except ImportError:
    pass  # Optional environment
