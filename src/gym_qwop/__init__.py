"""QWOP Gymnasium environment package."""
# Registers qwop-v0, qwop-realtime-v0, and qwop-authentic-v0 - see envs/__init__.py.
from gym_qwop.envs import QWOPEnv, QWOPRealtimeEnv, QWOPAuthenticEnv

__all__ = ['QWOPEnv', 'QWOPRealtimeEnv', 'QWOPAuthenticEnv']