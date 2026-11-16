"""Authentic QWOP environment: the actual, unmodified game, scheduled by the browser's
own native requestAnimationFrame - not ws.js's hijacked single-slot callback store
(used by QWOPEnv's lock-step driving and QWOPRealtimeEnv's setInterval-approximated
real-time driving). Actions are applied via genuine Selenium ActionChains key events,
landing in QWOP.min.js's own native key-handling pipeline exactly the way a human's
keystrokes would - see game/public/ws_authentic.js for the minimal (~15 line) hook this
env depends on, and its module comment for why it exists as a separate file from ws.js
rather than a flag on it.

This is not a training or fast-iteration environment - it's specifically for the final,
trustworthy step: verifying a trained policy's performance against a genuine wall-clock
duration (self._episode_wall_start-based, see step()'s info dict), with zero timing
approximation anywhere in the loop, for comparison against a human world-record time.
Use QWOPEnv for training and QWOPRealtimeEnv for fast iterative real-time-ish eval; use
this only for runs you want to actually trust the reported time of.
"""
import time
from collections import deque
from typing import Dict, Tuple, Any, Optional
import numpy as np
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException

from .qwop_selenium_base import SeleniumQWOPEnvBase

import logging
logger = logging.getLogger(__name__)

_KEY_NAMES = ["q", "w", "o", "p"]  # index order matches _decode_action's bit order


class QWOPAuthenticEnv(SeleniumQWOPEnvBase):
    """
    Real QWOP environment running with the game's own native requestAnimationFrame
    scheduling, untouched - genuinely browser-paced physics, not an approximation of it.
    """

    def __init__(
        self,
        render_mode="human",
        render_fps=60,
        url: str = None,
        timeout: int = 10,
        enable_sound: bool = False,
        auto_start_server: bool = True,
        server_path: Optional[str] = None,
        server_port: Optional[int] = None,
        speed_reward_coef: float = 0.0,
        posy_reward_coef: float = 0.0,
        stride_reward_coef: float = 1.0,
        win_reward_coef: float = 0.0,
        require_footstep: bool = True,
        action_mode: str = "categorical",
        include_key_state: bool = True,
        min_ticks_per_step: int = 1,
    ):
        """
        Initialize the authentic QWOP environment.

        Args:
            render_mode: Defaults to "human" (unlike the other envs, which default to
                None/headless) - SeleniumQWOPEnvBase sets headless = render_mode not in
                ("human", "realtime"), and this env is specifically for runs you want
                to trust, where there's no reason to hide the browser and no training
                throughput to protect. Pass render_mode=None explicitly to run headless
                anyway, though genuine requestAnimationFrame firing reliably in a
                headless context is untested here - verify before relying on it.
            min_ticks_per_step: Minimum number of NEW physics ticks that must land
                before step() returns, waited for via a single execute_async_script
                call (window.__qwopWaitForTicks in ws_authentic.js) - same idea as
                QWOPRealtimeEnv's parameter of the same name, and just as necessary
                here: without it, a fast enough Python-side round trip (no neural
                network inference, or just quick Selenium calls) can easily land
                within the same ~33ms physics tick as the previous read, forcing
                sim_dt=0 and therefore velx/every joint velocity to be reported as
                exactly zero even while the character is actively moving - confirmed
                empirically (ticks_this_step was 0 a majority of the time before this
                existed). Doesn't pace or throttle anything - physics keeps advancing
                on its own regardless - it only puts a floor on how little sim time a
                single step() can cover, exactly like QWOPRealtimeEnv's version.

            (See SeleniumQWOPEnvBase.__init__ for url/timeout/enable_sound/
            auto_start_server/server_path/server_port, the reward coefficients
            posy_reward_coef/stride_reward_coef/speed_reward_coef/win_reward_coef,
            require_footstep, and action_mode/include_key_state. No ticks_per_step or
            speed_multiplier here - there's no artificial per-step tick count or speed
            control to set; physics runs at exactly its own natural pace and step()
            acts then waits only as long as min_ticks_per_step requires, the same way
            a human's reaction time - not an artificial floor - is what bounds their
            input rate.)

        Environment variables:
            QWOP_URL: Override server URL
            QWOP_SERVER_PATH: Path to game server
            QWOP_HEADLESS: Run headless (true/false)
            QWOP_TIMEOUT: Page load timeout in seconds
            QWOP_SERVER_PORT: Default port for game server
        """
        self.min_ticks_per_step = min_ticks_per_step
        # Pipeline-latency instrumentation (see _measure_action_latency): ticks between
        # the observation a decision was made from and the tick its resulting key change
        # actually took effect. _step_count throttles how often the on-page HUD gets
        # updated (see _apply_action_and_read) - every step would add a Selenium round
        # trip of its own on top of the very latency being measured.
        self._latency_history = deque(maxlen=50)
        self._last_action_latency_ticks = None
        self._step_count = 0

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
        # land - give it real headroom beyond the nominal ~33ms/tick wait (Selenium's
        # default script timeout can be too short once round-trip/GC jitter is added).
        self.driver.set_script_timeout(10.0)

    @property
    def _URL_QUERY(self) -> str:
        return "webgl=true&authentic=1"

    def _wait_for_physics_ready(self):
        """Same reasoning as QWOPRealtimeEnv: the game's own native
        requestAnimationFrame loop is already running from page load (rendering at
        least), so there's nothing to pump - just poll for window.state to become the
        real data object rather than the DOM auto-binding stub. Physics ticking itself
        (window.__qwopStepWorld) still doesn't start until the click-to-begin gesture
        in reset() - see QWOP.min.js's firstClick gate - but that doesn't block this
        check, since window.state gets populated by the render loop regardless."""
        deadline = time.time() + self.timeout + 5
        while True:
            if self.driver.execute_script(
                "return !!(window.state && typeof window.state.posx !== 'undefined');"
            ):
                return
            if time.time() >= deadline:
                raise TimeoutException("Game physics loop did not start in time")
            time.sleep(0.02)

    # Waits for at least min_ticks_per_step NEW physics ticks (window.__qwopWaitForTicks
    # in ws_authentic.js) before resolving with the current state - no action-application
    # coupling here (unlike QWOPRealtimeEnv's equivalent), since actions go through
    # genuine ActionChains in a separate call, not a JS-dispatched helper.
    _WAIT_JS = (
        "var minTicks = arguments[0], callback = arguments[arguments.length - 1];"
        "window.__qwopWaitForTicks(minTicks, callback);"
    )

    @staticmethod
    def _measure_action_latency(t_obs: int, target_keys, key_log) -> Optional[int]:
        """Pipeline latency, in ticks: how long after the observation at tick t_obs
        (the one this step's action was decided from) does the game's own native key
        state (window.game.QDown/WDown/ODown/PDown, logged every tick by
        ws_authentic.js - not anything this class dispatches or reads directly) first
        match target_keys, the full 4-key state just sent via ActionChains.

        1 tick is the theoretical floor, not 0: the observation at t_obs was already
        final by the time Python received it, and physics keeps ticking natively the
        entire time Python spends deciding and dispatching the action - there's no way
        for that dispatch to land within the same tick it was decided from. Anything
        above 1 is round-trip/inference overhead beyond that floor. Returns None if no
        matching entry is found in the (bounded, most-recent-64-ticks) log - either the
        round trip is unusually slow, or the log window has already rolled past it.
        """
        target = tuple(bool(k) for k in target_keys)
        for tick, q, w, o, p in key_log:
            if tick >= t_obs and (q, w, o, p) == target:
                return tick - t_obs
        return None

    def _update_latency_hud(self):
        """Push the current rolling-average latency to the on-page HUD (#action-latency
        in index.html) - throttled (see _step_count) so this doesn't add a Selenium
        round trip on every single step, which would itself distort the very latency
        being measured."""
        if not self._latency_history:
            return
        avg = sum(self._latency_history) / len(self._latency_history)
        text = f"Pipeline latency: {avg:.2f} ticks (n={len(self._latency_history)})"
        self.driver.execute_script(
            "var el = document.getElementById('action-latency'); "
            "if (el) { el.style.display = 'block'; el.innerText = arguments[0]; }",
            text,
        )

    def _apply_action_and_read(self, keys: Optional[list]) -> Dict[str, float]:
        """Apply only the keys that actually changed since last call, via genuine
        ActionChains key events (landing in QWOP.min.js's own native on_keydown/
        on_keyup handlers - see the module docstring), then wait for at least
        min_ticks_per_step new ticks and read the resulting state. Physics advances
        entirely on its own between calls (real requestAnimationFrame, never paused or
        driven by this call) - this is purely "act, then wait/observe", not "advance n
        ticks" the way QWOPEnv's _advance_and_get_state is."""
        try:
            t_obs = self.last_tick  # the observation tick this decision was made from
            changed_now = False
            if keys is not None:
                changed = np.asarray(keys, dtype=np.float32) != self._current_keys
                if changed.any():
                    changed_now = True
                    chain = ActionChains(self.driver)
                    for i, key_name in enumerate(_KEY_NAMES):
                        if not changed[i]:
                            continue
                        if keys[i]:
                            chain.key_down(key_name)
                        else:
                            chain.key_up(key_name)
                    chain.perform()
                self._current_keys = np.asarray(keys, dtype=np.float32)
            raw = self.driver.execute_async_script(self._WAIT_JS, self.min_ticks_per_step)

            if changed_now:
                latency = self._measure_action_latency(t_obs, self._current_keys, raw.get("keyLog", []))
                if latency is not None:
                    self._last_action_latency_ticks = latency
                    self._latency_history.append(latency)

            self._step_count += 1
            if self._step_count % 15 == 0:
                self._update_latency_hud()

            return self._compute_state(raw)
        except Exception as e:
            logger.error(f"Error applying action/reading game state: {e}")
            raise RuntimeError(f"Failed to apply action/read game state: {e}")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step: apply whatever keys changed, then wait for at
        least min_ticks_per_step new ticks and read the result. Control frequency is
        otherwise bounded only by how fast Python's own decision loop and the Selenium
        round trip run, same as a human's reaction time bounds theirs - min_ticks_per_step
        just guards against a same-tick, zero-progress read (see its docstring)."""
        keys = self._decode_action(action)
        state = self._apply_action_and_read(keys)
        obs = self.obs(state)
        reward = self.compute_reward(state)

        # state["won"] (window.state.won, set mid-tick in QWOP.min.js's own per-tick
        # state computation) can go True a tick before window.terminal does - endGame()
        # sets window.terminal only after this tick's physics step already resolved the
        # pending __qwopWaitForTicks() wait with the stale value. Worse, endGame() also
        # sets this.pause = true on a win, which permanently stops QWOP.min.js from ever
        # calling window.__qwopStepWorld again (see QWOP.min.js:1043) until reset() - so
        # if termination were missed here, the *next* step()'s wait would hang forever
        # (10s Selenium timeout) against a game loop that will never tick again. Checking
        # won here (not just terminal) catches the win on the same step it happens,
        # before any further step() call can be issued against the paused loop.
        terminated = state.get("terminal", False) or state.get("won", False)
        truncated = False  # No timeout here - wrap externally (e.g. gym.make(max_episode_steps=...)) if desired.

        return obs, reward, terminated, truncated, {
            "state": state,
            "posx": float(state.get("posx", 0)),
            "velx": float(state.get("velx", 0)),
            "stride_count": self._episode_stride_count,
            "episode_sim_time": self._episode_sim_time,
            # The authoritative duration for this env - genuine time.time()-measured
            # wall-clock seconds, not derived from any tick count or JS-side timing
            # convention at all. This is the number to compare against a world record.
            "episode_wall_time": time.time() - self._episode_wall_start,
            "won": bool(state.get("won", False)),
            "ticks_this_step": self._last_ticks_elapsed,
            # Pipeline latency in ticks (see _measure_action_latency) - None on a step
            # where no key actually changed (nothing to measure) or where the round trip
            # was slow enough to fall outside ws_authentic.js's logging window.
            "action_latency_ticks": self._last_action_latency_ticks,
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
            # Genuine wall-clock start - see step()'s "episode_wall_time".
            self._episode_wall_start = time.time()
            # Goes through the same min_ticks_per_step wait as step() (see
            # _apply_action_and_read) - ensures the first observation reflects the
            # post-reset run, not a stale or zero-progress pre-reset read.
            state = self._apply_action_and_read(keys=None)
            self.last_frame_time = time.time()

            return self.obs(state), {}

        except Exception as e:
            logger.error(f"Error during reset: {e}")
            raise RuntimeError(f"Failed to reset game: {e}")
