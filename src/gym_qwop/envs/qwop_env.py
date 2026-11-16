"""Lock-step QWOP environment: physics only advances when explicitly driven from
Python (see window.__qwopAdvance in ws.js), never on its own - an exact, deterministic
action-repeat/frame-skip. See qwop_realtime_env.py for the alternative where physics
advances on its own, independent of the agent."""
import time
from typing import Dict, Tuple, Any, Optional
import numpy as np
from selenium.webdriver.common.action_chains import ActionChains

from .qwop_selenium_base import SeleniumQWOPEnvBase

import logging
logger = logging.getLogger(__name__)


class QWOPEnv(SeleniumQWOPEnvBase):
    """
    Real QWOP environment using Selenium browser automation, with physics lock-stepped
    to the agent's own actions.

    Configuration via environment variables:
    - QWOP_URL: Game server URL (default: http://localhost:3000)
    - QWOP_HEADLESS: Run browser in headless mode (default: true)
    - QWOP_TIMEOUT: Page load timeout in seconds (default: 10)
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
        ticks_per_step: int = 1,
        speed_reward_coef: float = 0.0,
        posy_reward_coef: float = 0.0,
        stride_reward_coef: float = 1.0,
        win_reward_coef: float = 0.0,
        require_footstep: bool = True,
        action_mode: str = "categorical",
        include_key_state: bool = True,
    ):
        """
        Initialize real QWOP environment.

        Args:
            ticks_per_step: Number of physics ticks (each PHYSICS_DT simulated seconds)
                to run per env.step()/reset() call. Physics is lock-stepped to the
                agent - it only advances when explicitly driven (see
                window.__qwopAdvance in ws.js), not on a real-time timer - so this is an
                exact, deterministic action-repeat/frame-skip rather than depending on
                Selenium round-trip timing. Raise it to cover more simulated time per RL
                step (and, since ticks run synchronously in one script call, to train
                faster).

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
        self.ticks_per_step = ticks_per_step

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

    # Combined read of window.state / window.terminal / window.physicsTick in one round
    # trip. Optionally applies an action (window.__qwopApplyAction, JS-dispatched key
    # events - no separate ActionChains round trip) and/or advances physics by `ticks`
    # first, still all in the same round trip - so step()/reset() each need only one
    # execute_script call total instead of a key-press call, an advance call, and three
    # separate reads.
    _ADVANCE_AND_READ_JS = "window.__qwopAdvance(arguments[0]);" + SeleniumQWOPEnvBase._READ_RAW_STATE_JS
    _STEP_JS = "window.__qwopApplyAction(arguments[0]); window.__qwopAdvance(arguments[1]);" + SeleniumQWOPEnvBase._READ_RAW_STATE_JS

    def _advance_and_get_state(self, ticks: int, action: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Advance physics by `ticks` and read the resulting state, optionally applying an
        action first - all in a single round trip rather than a separate action-application
        call, a separate __qwopAdvance call, and three separate reads."""
        try:
            if action is not None:
                raw = self.driver.execute_script(self._STEP_JS, np.asarray(action).tolist(), ticks)
            else:
                raw = self.driver.execute_script(self._ADVANCE_AND_READ_JS, ticks)
            return self._compute_state(raw)
        except Exception as e:
            logger.error(f"Error advancing/getting game state: {e}")
            raise RuntimeError(f"Failed to advance game state: {e}")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step: apply the action, advance physics by exactly
        ticks_per_step (see PHYSICS_DT / __qwopAdvance), then observe the result - all in
        one round trip (see _advance_and_get_state)."""
        keys = self._decode_action(action)
        # The keys just applied stay held until the next call changes them - so they're
        # exactly the key state this step's resulting obs() should report (see base.py's
        # _current_keys comment).
        self._current_keys = np.asarray(keys, dtype=np.float32)
        state = self._advance_and_get_state(self.ticks_per_step, action=keys)
        obs = self.obs(state)
        reward = self.compute_reward(state)

        # No-op unless actually rendering (render_mode "human" or "realtime"); see
        # BaseQWOPEnv._throttle. Passing the actual simulated time this step covered makes
        # "realtime" mode's playback speed correct regardless of ticks_per_step.
        self._throttle(self.ticks_per_step * self.PHYSICS_DT)

        terminated = state.get("terminal", False)
        truncated = False  # No timeout in real environment

        # "posx" is also exposed as a plain top-level scalar (in addition to the full
        # "state" dict) because gymnasium's vector-env info merging recursively decomposes
        # nested dicts field-by-field - a scalar is far simpler to read back out for a
        # specific env index than re-navigating that recursive structure.
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
            state = self._advance_and_get_state(self.ticks_per_step)
            self.last_frame_time = time.time()

            return self.obs(state), {}

        except Exception as e:
            logger.error(f"Error during reset: {e}")
            raise RuntimeError(f"Failed to reset game: {e}")
