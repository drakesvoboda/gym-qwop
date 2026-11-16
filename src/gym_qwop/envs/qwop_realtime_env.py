"""Real-time QWOP environment: physics advances on its own, via an autonomous JS timer
independent of Python (see the __qwopRealtime block in ws.js) - the agent cannot pause
or influence how much sim time passes between its actions, unlike QWOPEnv's lock-step
model. Intended primarily for evaluating a trained policy under fair, human-comparable
timing (e.g. benchmarking against a world-record time); see qwop_env.py for the fast,
deterministic environment used for actual training.
"""
import time
from typing import Dict, Tuple, Any, Optional
import numpy as np
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException

from .qwop_selenium_base import SeleniumQWOPEnvBase

import logging
logger = logging.getLogger(__name__)


class QWOPRealtimeEnv(SeleniumQWOPEnvBase):
    """
    Real QWOP environment using Selenium browser automation, with physics running on
    its own real-time (or faster-than-real-time) clock, independent of the agent.
    """

    def __init__(
        self,
        render_mode=None,
        render_fps=60,
        url: str = None,
        timeout: int = 10,
        enable_sound: bool = False,
        auto_start_server: bool = True,
        server_path: Optional[str] = None,
        server_port: Optional[int] = None,
        speed_multiplier: float = 1.0,
        min_ticks_per_step: int = 1,
        speed_reward_coef: float = 0.0,
        posy_reward_coef: float = 0.0,
        stride_reward_coef: float = 1.0,
        win_reward_coef: float = 0.0,
        require_footstep: bool = True,
        action_mode: str = "categorical",
        include_key_state: bool = True,
    ):
        """
        Initialize real-time QWOP environment.

        Args:
            speed_multiplier: How much faster than true wall-clock the autonomous
                physics timer runs (ws.js: setInterval(tick, FRAME_INTERVAL_MS /
                speed_multiplier)). 1.0 (default) is genuine real-time - use this for
                evaluating a policy under fair, human-comparable timing (e.g.
                benchmarking against a world-record time). Values > 1 "overclock" the
                environment for faster training throughput while still preserving the
                core property that the agent doesn't control the pace: ticks keep
                accruing on the independent timer regardless of how long Python takes
                between step() calls, just at a faster clock rate.
            min_ticks_per_step: Minimum number of NEW physics ticks that must have
                elapsed before step() returns, waited for via a single
                execute_async_script call (window.__qwopWaitForTicks) rather than
                Python-side polling - avoids a near-zero-progress read (velx≈0) if
                step() happens to be called faster than physics is currently ticking.
                This doesn't let the agent go faster than the timer's real pace; it
                only puts a floor on how little sim time a single step() can cover.

            (See SeleniumQWOPEnvBase.__init__ for url/timeout/enable_sound/
            auto_start_server/server_path/server_port, the reward coefficients
            posy_reward_coef/stride_reward_coef/speed_reward_coef/win_reward_coef,
            require_footstep, and action_mode/include_key_state.)

        Environment variables:
            QWOP_URL: Override server URL
            QWOP_SERVER_PATH: Path to game server
            QWOP_HEADLESS: Run headless (true/false)
            QWOP_TIMEOUT: Page load timeout in seconds
            QWOP_SERVER_PORT: Default port for game server
        """
        # Set before super().__init__() - _connect_to_game() (called from there) reads
        # self._URL_QUERY, which depends on speed_multiplier.
        self.speed_multiplier = speed_multiplier
        self.min_ticks_per_step = min_ticks_per_step

        super().__init__(
            render_mode=render_mode,
            render_fps=render_fps,
            url=url,
            timeout=timeout,
            enable_sound=enable_sound,
            auto_start_server=auto_start_server,
            server_path=server_path,
            server_port=server_port,
            posy_reward_coef=posy_reward_coef,
            stride_reward_coef=stride_reward_coef,
            speed_reward_coef=speed_reward_coef,
            win_reward_coef=win_reward_coef,
            require_footstep=require_footstep,
            action_mode=action_mode,
            include_key_state=include_key_state,
        )

        # execute_async_script blocks (JS-side) until at least min_ticks_per_step ticks
        # land - give it real headroom beyond the nominal wait time (Selenium's default
        # script timeout can be too short once round-trip/GC jitter is added on top).
        nominal_wait_s = self.min_ticks_per_step * self.PHYSICS_DT / self.speed_multiplier
        self.driver.set_script_timeout(max(10.0, nominal_wait_s * 5))

    @property
    def _URL_QUERY(self) -> str:
        return f"webgl=true&realtime=1&speed={self.speed_multiplier}"

    _WAIT_JS = (
        "var action = arguments[0], minTicks = arguments[1], callback = arguments[arguments.length - 1];"
        "window.__qwopWaitForTicks(action, minTicks, callback);"
    )

    def _wait_for_physics_ready(self):
        """The autonomous timer (see ws.js's __qwopRealtime block) is already ticking
        from page load - unlike lock-step, there's nothing to pump, just poll for
        window.state to become the real data object rather than the DOM auto-binding
        stub (see SeleniumQWOPEnvBase._wait_for_physics_ready's docstring for why
        element presence alone isn't enough)."""
        deadline = time.time() + self.timeout + 5
        while True:
            if self.driver.execute_script(
                "return !!(window.state && typeof window.state.posx !== 'undefined');"
            ):
                return
            if time.time() >= deadline:
                raise TimeoutException("Game physics loop did not start in time")
            time.sleep(0.02)

    def _apply_and_wait(self, action: Optional[np.ndarray] = None, min_ticks: Optional[int] = None) -> Dict[str, float]:
        """Apply `action` (if given) and block until at least `min_ticks` new physics
        ticks have landed, all in one execute_async_script round trip - see
        window.__qwopWaitForTicks in ws.js. Physics itself advances independently of
        this call (the autonomous timer), so this is purely "wait and read", not
        "drive n ticks" the way QWOPEnv._advance_and_get_state is."""
        min_ticks = self.min_ticks_per_step if min_ticks is None else min_ticks
        try:
            action_arg = np.asarray(action).tolist() if action is not None else None
            raw = self.driver.execute_async_script(self._WAIT_JS, action_arg, min_ticks)
            return self._compute_state(raw)
        except Exception as e:
            logger.error(f"Error waiting/getting game state: {e}")
            raise RuntimeError(f"Failed to wait for game state: {e}")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step: apply the action, then wait for physics
        (already ticking on its own) to have advanced by at least min_ticks_per_step
        before observing the result."""
        keys = self._decode_action(action)
        self._current_keys = np.asarray(keys, dtype=np.float32)
        state = self._apply_and_wait(keys)
        obs = self.obs(state)
        reward = self.compute_reward(state)

        # No _throttle() call here, unlike QWOPEnv - it would be redundant. Physics
        # already runs on its own real-time (or speed_multiplier-scaled) clock, and
        # _apply_and_wait() blocks until that clock has actually produced
        # min_ticks_per_step new ticks, so by the time step() returns, the correct
        # amount of real wall-clock time has already elapsed on its own. speed_multiplier
        # is the one authoritative pacing control for this env - render_mode="realtime"'s
        # _throttle mechanism exists for QWOPEnv, which (being lock-step) has no
        # inherent real-time pacing of its own to rely on.
        terminated = state.get("terminal", False)
        truncated = False  # No timeout in real environment

        return obs, reward, terminated, truncated, {
            "state": state,
            "posx": float(state.get("posx", 0)),
            "velx": float(state.get("velx", 0)),
            "stride_count": self._episode_stride_count,
            "episode_sim_time": self._episode_sim_time,
            "won": bool(state.get("won", False)),
            "ticks_this_step": self._last_ticks_elapsed,
        }

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset game to initial state."""
        try:
            self._lead_side = 0
            self._pending_side = 0
            self._pending_sim_time = 0.0
            self._pending_foot_down = False
            self._episode_stride_count = 0
            self._episode_sim_time = 0.0

            self.canvas.click()
            ActionChains(self.driver) \
                .key_up("q") \
                .key_up("w") \
                .key_up("o") \
                .key_up("p") \
                .key_down("r") \
                .key_up("r") \
                .perform()

            self.old_posx = 0.0
            self.old_velx = 0.0
            self.last_joint_angles = {}
            self._current_keys = np.zeros(4, dtype=np.float32)  # all keys released above
            # No action to apply here (None), just wait for at least one fresh tick so
            # the first observation reflects the post-reset run, not stale pre-reset state.
            state = self._apply_and_wait(action=None, min_ticks=1)
            self.last_frame_time = time.time()

            return self.obs(state), {}

        except Exception as e:
            logger.error(f"Error during reset: {e}")
            raise RuntimeError(f"Failed to reset game: {e}")
