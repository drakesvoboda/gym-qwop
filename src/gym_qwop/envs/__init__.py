"""QWOP Gymnasium environments."""
import gymnasium as gym
from gymnasium.envs.registration import register

from gym_qwop.envs.qwop_env import QWOPEnv
from gym_qwop.envs.qwop_realtime_env import QWOPRealtimeEnv
from gym_qwop.envs.qwop_authentic_env import QWOPAuthenticEnv

# Register only if not already registered
def _register_env(id, entry_point):
    if id not in gym.registry:
        register(id=id, entry_point=entry_point)

# Real QWOP (requires game server + Selenium + Firefox), physics lock-stepped to the agent
_register_env('qwop-v0', 'gym_qwop.envs.qwop_env:QWOPEnv')

# Real QWOP with physics running on its own (real-time or faster), independent of the
# agent - see qwop_realtime_env.py. Primarily for evaluating a trained policy under fair,
# human-comparable timing rather than for training.
_register_env('qwop-realtime-v0', 'gym_qwop.envs.qwop_realtime_env:QWOPRealtimeEnv')

# Real QWOP with the game's own native requestAnimationFrame scheduling, untouched - not
# an approximation of real-time pacing (unlike qwop-realtime-v0) but the genuine thing.
# See qwop_authentic_env.py. For final, trustworthy world-record-comparison runs only.
_register_env('qwop-authentic-v0', 'gym_qwop.envs.qwop_authentic_env:QWOPAuthenticEnv')


__all__ = ['register', 'QWOPEnv', 'QWOPRealtimeEnv', 'QWOPAuthenticEnv']