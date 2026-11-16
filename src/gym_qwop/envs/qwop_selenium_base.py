"""Shared base for the Selenium-driven QWOP env variants (lock-step and real-time).

Everything here is timing-model-agnostic: game connection/startup, state reading,
reward shaping, and the stride-detection state machine all work identically regardless
of how physics ticks get advanced. What genuinely differs between variants - how
step()/reset() drive ticks - stays in each subclass (qwop_env.py's QWOPEnv for
lock-step, qwop_realtime_env.py's QWOPRealtimeEnv for real-time).
"""
import os
import time
import math
import signal
import atexit
import logging
from typing import Dict, Tuple, Any, Optional
import numpy as np
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from .base import BaseQWOPEnv

logger = logging.getLogger(__name__)


class SeleniumQWOPEnvBase(BaseQWOPEnv):
    """
    Shared connection/state/reward logic for Selenium-driven QWOP environments.

    Configuration via environment variables:
    - QWOP_URL: Game server URL (default: http://localhost:3000)
    - QWOP_HEADLESS: Run browser in headless mode (default: true)
    - QWOP_TIMEOUT: Page load timeout in seconds (default: 10)
    """

    # Fixed simulated seconds per physics tick (matches world.step(0.04, ...) in
    # QWOP.min.js) - Box2D's own physics-integration timestep. Used for velx/accelx/
    # joint velocities/STRIDE_MIN_DWELL_SEC, where what matters is a self-consistent
    # internal unit, not real-world calibration.
    PHYSICS_DT = 0.04

    # How many REAL/human seconds one physics tick represents - a completely separate
    # constant from PHYSICS_DT, matching QWOP.min.js's own game-loop accumulator
    # constant (`frame_time = 0.03333...`, QWOP.min.js:506), which is what the game's
    # own scoreTime ("X metres in Y seconds") actually accumulates by. PHYSICS_DT and
    # this constant are NOT the same thing and were never meant to be: world.step()'s
    # 0.04 is a physics-solver tuning choice, independent of the game loop's separate
    # real-time-tracking design (a classic fixed-timestep accumulator, which is what
    # makes scoreTime track genuine wall-clock time regardless of actual render
    # framerate) - the two constants just happen to both fire once per tick in the
    # same update() call. Used only where something is meant to be human/world-record
    # comparable - see _episode_sim_time below - never for velx/physics purposes.
    REAL_SECONDS_PER_TICK = 1.0 / 30.0

    # Deadzone half-width (radians) around stride≈0 that counts as "ambiguous" rather
    # than clearly left- or right-leading - avoids chattering swap detection right at
    # the midline. Held keys apply continuous torque (thigh angles aren't bounded to a
    # small range - they can spin past +/-pi), so the lead side is read off
    # wrap_to_pi(leftthigh - rightthigh), not the raw angle difference.
    STRIDE_SWAP_THRESHOLD = 0.5
    # Sim-seconds the new side must hold past the deadzone before a swap counts, so a
    # brief twitch across the midline doesn't register as a real stride. Sim-time
    # (accumulated from actual elapsed ticks), not a step count - a step count is only a
    # consistent amount of time under lock-step's fixed ticks_per_step; under the
    # real-time env, where ticks-per-step genuinely varies, "N steps" would be a wildly
    # inconsistent dwell requirement (sometimes a fraction of a real gait-cycle beat,
    # sometimes several), making swap detection unreliable. 0.6s matches the original
    # calibration (5 steps * ticks_per_step=3 * PHYSICS_DT=0.04 under lock-step).
    STRIDE_MIN_DWELL_SEC = 0.6

    # Query string appended to the game URL - subclasses override to select ws.js's
    # timing mode (see ws.js's `__qwopRealtime` flag). Lock-step is the default here;
    # QWOPRealtimeEnv overrides this to add "&realtime=1".
    _URL_QUERY = "webgl=true"

    # Combined read of window.state / window.terminal / window.physicsTick in one round
    # trip - shared by both variants' step()/get_state() JS snippets.
    _READ_RAW_STATE_JS = "return {state: window.state || null, terminal: !!window.terminal, tick: window.physicsTick || 0};"

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
        posy_reward_coef: float = 0.0,
        stride_reward_coef: float = 1.0,
        speed_reward_coef: float = 0.0,
        win_reward_coef: float = 0.0,
        require_footstep: bool = True,
        action_mode: str = "categorical",
        include_key_state: bool = True,
    ):
        """
        Args:
            url: Game server URL (uses QWOP_URL env var if not provided)
            headless: Run browser in headless mode
            timeout: Page load timeout
            enable_sound: Enable browser audio
            auto_start_server: Automatically start game server if not running
            server_path: Path to game server (auto-detect if None)
            server_port: Port for game server (falls back to QWOP_SERVER_PORT env
                var, then 3100). Each parallel instance needs its own port.
            posy_reward_coef: Weight on a standing-height *bonus*, added on top of the
                forward-velocity reward: coef * max(0, -posy). The engine's world y-axis
                points *down* (gravity is (0, +10), posy is the runner's world-center y),
                so upright posture is negative posy (measured empirically: ~-0.18
                standing, ~-0.03 settled in a crouch, ~+0.5 at a face-plant's terminal
                tick) - max(0, -posy) is positive while upright and floors at exactly 0
                once posy crosses into "on the ground" territory, rather than going
                negative. Deliberately a bonus, not a penalty: a per-step penalty for low
                posture creates a flow cost the agent can escape by ending the episode
                (falling) rather than recovering, since falling zeroes out all future
                reward instead of accumulating more of the penalty. A floored bonus never
                produces that incentive - the worst outcome is forfeiting the bonus, not
                paying anything. Set to 0 to disable.
            stride_reward_coef: Weight on a bonus paid the instant the "lead leg"
                (whichever thigh is more forward, from wrap_to_pi(leftthigh -
                rightthigh)) actually swaps sides - see _stride_bonus. Swap detection
                uses a deadzone (STRIDE_SWAP_THRESHOLD) around stride≈0 plus a minimum
                dwell time on the new side (STRIDE_MIN_DWELL_SEC sim-seconds) before it
                counts, so briefly twitching across the midline doesn't register as a
                swap. The
                very first side the policy settles into after reset isn't rewarded
                (there's nothing to have swapped from yet). Set to 0 to disable.
            speed_reward_coef: Weight on the dense per-tick forward-speed term
                (velx * abs(velx), sign-preserving so backward motion isn't rewarded).
                Set to 0 to make all forward-progress credit gated behind stride
                completion instead (see stride_reward_coef).
            win_reward_coef: Weight on a one-time bonus paid on the terminal tick where
                state["won"] is True (reaching the 100m finish - see window.state.won,
                set from QWOP.min.js's jumpLanded): coef / episode_sim_time. Naturally
                pays out exactly once: winning ends the episode (see the win-aware
                bootstrap handling in rlqwop.PPO), so there's no dwell-time/repeat-
                prevention logic to worry about the way _stride_bonus needs. None of
                the other reward terms actually reward *finishing fast* - they reward
                moving well en route, which is only a proxy - so without this, a
                58-second finish and a 48-second finish get identical credit as long as
                both actually finish.

                Inverse-time rather than a flat constant or a fixed-target-relative
                bonus (an earlier version of this used max(0, target - episode_sim_time),
                which needed a hand-tuned, manually-lowered-over-time target): 1/t is
                always positive (never a penalty for a slow-but-real finish, matching
                this class's bonus-not-penalty philosophy - see posy_reward_coef above),
                needs no target hyperparameter at all, and is convex in t - the marginal
                reward for shaving one more second (d(bonus)/dt = -coef/t^2) grows as t
                shrinks, so the same 5-second improvement is worth more once already
                finishing quickly than it is early in training. Set to 0 to disable
                (the default - every other coefficient here defaults to something that
                already produces a reasonable training signal without it).

                Calibrating coef: aim for coef to land near the *total* dense reward a
                typical episode already accumulates - i.e. coef ~= EpAvgVelx^2 *
                (typical episode length in RL steps) - big enough to be felt (GAE with
                gamma/lam effectively smears the terminal reward back over roughly
                1/(1-gamma*lam) steps, not just the last tick), small enough not to
                swamp the existing dense signal that already produced a working policy.
                This is deliberately independent of episode_sim_time's units (T cancels
                out of coef/T * T_per_step * n_steps algebraically, leaving just
                velx^2 * n_steps) - read EpAvgVelx and typical episode length straight
                from your own training logs rather than assuming a value, since those
                are unaffected by anything above.
            require_footstep: If True (default), a stride only confirms once the newly-
                lead leg has actually planted on the ground (window.leftFootDown/
                rightFootDown, set from Box2D contact events in QWOP.min.js) at some
                point during its dwell window, not just once the thigh angle alone has
                crossed the swap threshold for long enough - rules out a leg swinging
                forward in the air counting as a completed stride. Set to False to fall
                back to dwell-time-only detection (the original behavior, before
                foot-contact tracking existed).
            action_mode/include_key_state: See BaseQWOPEnv.__init__.

        Environment variables:
            QWOP_URL: Override server URL
            QWOP_SERVER_PATH: Path to game server
            QWOP_HEADLESS: Run headless (true/false)
            QWOP_TIMEOUT: Page load timeout in seconds
            QWOP_SERVER_PORT: Default port for game server
        """

        super().__init__(render_mode, render_fps, action_mode=action_mode, include_key_state=include_key_state)

        self.posy_reward_coef = posy_reward_coef
        self.stride_reward_coef = stride_reward_coef
        self.speed_reward_coef = speed_reward_coef
        self.win_reward_coef = win_reward_coef
        self.require_footstep = require_footstep

        # Which slot this instance occupies in a vectorized run, 0 by default (also
        # correct for a single, non-vectorized env) - mutated externally after
        # construction (see rlqwop.PPO.ppo's make_env) so render_value_estimate knows
        # which entry of a broadcast per-worker values list is its own.
        self.worker_index = 0

        # Lead-leg swap tracking (see _stride_bonus / stride_reward_coef docstring).
        # _lead_side is 0 (undetermined) until the policy first settles clearly to one
        # side; _pending_side/_pending_sim_time track a candidate swap-in-progress before
        # it's held long enough (STRIDE_MIN_DWELL_SEC) to count.
        self._lead_side = 0
        self._pending_side = 0
        self._pending_sim_time = 0.0
        self._pending_foot_down = False
        self._episode_stride_count = 0
        # Elapsed sim-time (seconds) / raw tick count of the most recent _compute_state
        # call - a fixed constant (ticks_per_step, and ticks_per_step * PHYSICS_DT)
        # under lock-step, but genuinely variable under real-time timing, where
        # _last_sim_dt is the actual pacing signal for _throttle() and
        # _last_ticks_elapsed is exposed to Python as a measure of the real effective
        # control granularity - i.e. how many ticks actually fire between one action
        # and the next.
        self._last_sim_dt = 0.0
        self._last_ticks_elapsed = 0
        # Cumulative elapsed real/human-equivalent seconds so far this episode -
        # accumulated via REAL_SECONDS_PER_TICK (not PHYSICS_DT - see that constant's
        # comment for why they're different), so this approximates the same clock a
        # real QWOP player (or the world-record stopwatch) would be on, regardless of
        # timing mode. Still an approximation under lock-step, where physics ticks
        # don't correspond to any real elapsed time at all (Python drives them
        # explicitly) - see QWOPAuthenticEnv for a mode that measures this genuinely
        # rather than approximating it from a tick count.
        self._episode_sim_time = 0.0

        if server_port is None:
            server_port = self._resolve_server_port(3100)

        self.url = url or os.getenv("QWOP_URL", f"http://localhost:{server_port}")
        self.headless = self.render_mode not in ("human", "realtime")
        self.timeout = timeout or int(os.getenv("QWOP_TIMEOUT", "10"))
        self.enable_sound = enable_sound

        self.server_manager = self._start_game_server(server_path, server_port, self.timeout) if auto_start_server else None

        self.canvas = None

        # For velocity tracking
        self.last_joint_angles = {}
        self.joint_names = ["head", "torso", "leftarm", "rightarm", "leftthigh", "rightthigh", "leftcalf", "rightcalf"]

        self._connect_to_game()

        # Setup signal handlers for graceful shutdown
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGTSTP]:
            signal.signal(sig, self._signal_handler)
        atexit.register(self.close)

    def _connect_to_game(self):
        """Connect to QWOP game server."""
        try:
            logger.info(f"Connecting to QWOP at {self.url}")

            self.driver = self._launch_browser(f"{self.url}?{self._URL_QUERY}", self.headless, self.enable_sound)
            logger.info("Page loaded in Firefox, waiting for game elements...")

            # Wait for game to load - wait for terminal element to appear and have content
            wait = WebDriverWait(self.driver, self.timeout + 5)
            # First wait for the static elements
            logger.debug("Waiting for terminal element...")
            wait.until(EC.presence_of_element_located((By.ID, "terminal")))
            logger.debug("Terminal element found, waiting for state element...")
            wait.until(EC.presence_of_element_located((By.ID, "state")))
            logger.debug("State element found, waiting for game canvas...")
            # Wait for the game canvas element (created by JavaScript)
            wait.until(EC.presence_of_element_located((By.ID, "window1")))
            logger.debug("Game canvas loaded, waiting for physics loop to be ready...")

            # Get game canvas element (modern API). State/terminal are read directly from
            # window.state / window.terminal in get_state() rather than the #state/#terminal
            # DOM elements, since QWOP.min.js sets both directly on window as well.
            self.canvas = self.driver.find_element(By.ID, "window1")

            self._wait_for_physics_ready()

            logger.debug("Game initialization complete")
            logger.info("QWOP game loaded successfully")
            # Initialize state trackers - will be populated on first step
            self.last_tick = self.driver.execute_script("return window.physicsTick || 0;")
            self.old_posx = 0
            self.old_velx = 0
            self.last_frame_time = time.time()

        except TimeoutException:
            logger.error(
                f"Game server not responding at {self.url}. "
                "Make sure the QWOP server is running.\n"
                "See: https://github.com/drakesvoboda/QWOP for setup instructions."
            )
            self.close()
            raise RuntimeError(f"QWOP server not responding at {self.url}")

    def _wait_for_physics_ready(self):
        """Block until window.state is the real data object, not the DOM auto-binding
        stub - see the lock-step subclass override for why this differs by timing mode.

        Default (lock-step): physics is lock-stepped to the agent (see
        window.__qwopAdvance in ws.js) - it doesn't run on its own timer, so nothing
        happens until we explicitly drive it. Element presence above isn't enough on
        its own either way: until the update loop has run at least once, `window.state`
        is still the browser's automatic named-element binding for <div id="state">
        (elements with an id are auto-exposed as window.<id>), not the real data object
        the game later assigns there. Use the plain one-shot pump here, not
        __qwopAdvance: physics can't actually tick yet (no click-to-start has
        happened), and __qwopAdvance would burn its whole retry budget waiting for a
        tick that can't happen."""
        deadline = time.time() + self.timeout + 5
        while True:
            self.driver.execute_script("window.__qwopPumpFrame();")
            if self.driver.execute_script(
                "return !!(window.state && typeof window.state.posx !== 'undefined');"
            ):
                return
            if time.time() >= deadline:
                raise TimeoutException("Game physics loop did not start in time")

    def _compute_state(self, raw: Dict[str, Any]) -> Dict[str, float]:
        """Turn a raw {state, terminal, tick} read from the browser into the full state
        dict, with diffx/velx/accelx derived from window.physicsTick (incremented once per
        simulated PHYSICS_DT-second world.step()) rather than wall-clock time. Since
        physics is lock-stepped to ticks_per_step (see step()/reset()), this is exact
        regardless of Selenium round-trip latency, GC pauses, or CPU contention from other
        parallel workers.
        """
        state = raw.get("state")

        # If state isn't available yet, return a default state
        if not state:
            logger.debug("window.state not yet available, using default state")
            return {
                "posx": 0,
                "quads": [0, 0, 0, 0],
                "terminal": False,
                "diffx": 0,
                "velx": 0,
                "accelx": 0,
            }

        state["terminal"] = bool(raw.get("terminal"))

        tick = raw.get("tick", 0)
        self._last_ticks_elapsed = tick - self.last_tick
        sim_dt = self._last_ticks_elapsed * self.PHYSICS_DT
        self.last_tick = tick
        self._last_sim_dt = sim_dt
        # REAL_SECONDS_PER_TICK, not sim_dt/PHYSICS_DT - see that constant's comment.
        self._episode_sim_time += self._last_ticks_elapsed * self.REAL_SECONDS_PER_TICK
        # Part of the observation (see base.py's obs()) - how many ticks the environment
        # just covered to reach this state. Under variable-cadence timing (the real-time
        # env) this is otherwise a hidden variable: the same visible state can result
        # from very different underlying dynamics depending on how much sim-time elapsed
        # to produce it, which the policy has no way to condition on without this.
        # Constant under lock-step, so harmless there.
        state["ticks_elapsed"] = self._last_ticks_elapsed

        state["diffx"] = state.get("posx", 0) - self.old_posx
        state["velx"] = state["diffx"] / sim_dt if sim_dt > 0 else 0.0
        state["accelx"] = (state["velx"] - self.old_velx) / sim_dt if sim_dt > 0 else 0.0

        self.old_posx = state.get("posx", 0)
        self.old_velx = state["velx"]

        # Joint angular velocities (part of the observation - see base.py's obs()),
        # finite-differenced the same way as velx above rather than read from Box2D
        # directly, since the game never exposes per-joint velocity itself. Safe to
        # diff the raw (unwrapped) angle directly - see observation_space's comment in
        # base.py's __init__ for why these angles never actually wrap even though
        # they're unbounded. self.last_joint_angles.get(joint, angle) defaults the
        # "previous" angle to the current one on the first call after __init__/reset
        # (see reset()'s self.last_joint_angles = {}), so that call reports 0 velocity
        # instead of a bogus spike from whatever pose preceded it.
        for joint in self.joint_names:
            angle = state.get(joint, 0.0)
            prev_angle = self.last_joint_angles.get(joint, angle)
            state[f"{joint}_vel"] = (angle - prev_angle) / sim_dt if sim_dt > 0 else 0.0
            self.last_joint_angles[joint] = angle

        return state

    def get_state(self) -> Dict[str, float]:
        """Get current game state, read directly from the window.state / window.terminal
        globals that QWOP.min.js sets on every physics tick, without advancing physics."""
        try:
            return self._compute_state(self.driver.execute_script(self._READ_RAW_STATE_JS))
        except Exception as e:
            logger.error(f"Error getting state: {e}")
            raise RuntimeError(f"Failed to get game state: {e}")

    def render_value_estimate(self, values_by_worker):
        """Push *this* worker's own critic value estimate to the #value-estimate HUD
        (index.html) - called identically on every worker via envs.call(...) (see
        rlqwop.PPO.ppo), which broadcasts the exact same values_by_worker list to every
        worker regardless of whether it's this process's SyncVectorEnv (num_envs=1) or
        a separate subprocess under AsyncVectorEnv (num_envs>1); each worker then reads
        only its own slot, indexed by worker_index (set once at construction - see
        ppo()'s make_env). This is how a per-worker (not shared/averaged) value display
        is possible at all without reaching into AsyncVectorEnv's internal pipe protocol
        for genuinely distinct per-worker calls, which isn't worth the risk of
        corrupting the vector env's state machine over a cosmetic display.
        """
        try:
            value = values_by_worker[self.worker_index]
        except (IndexError, TypeError):
            return
        self.driver.execute_script(
            "var el = document.getElementById('value-estimate'); "
            "if (el) { el.style.display = 'block'; el.innerText = arguments[0]; }",
            f"Critic V(s): {float(value):.2f} (worker {self.worker_index})",
        )

    def _stride_bonus(self, state: Dict[str, float], velx: float) -> float:
        """Pay stride_reward_coef * velx the moment the lead leg (whichever thigh is
        more forward) actually swaps sides AND that leg has planted on the ground at
        some point during the swap - see stride_reward_coef docstring in __init__.
        Requiring a footstep (not just a thigh angle crossing the swap threshold) rules
        out a leg swinging forward in the air counting as a completed stride - the swap
        only pays once the leg has actually taken weight. Stateful: call exactly once
        per real step (compute_reward's only caller), since it advances the dwell-time
        tracking each time it runs."""
        stride = state.get("leftthigh", 0.0) - state.get("rightthigh", 0.0)
        stride = (stride + math.pi) % (2 * math.pi) - math.pi  # wrap to (-pi, pi]

        if stride > self.STRIDE_SWAP_THRESHOLD:
            side = 1
        elif stride < -self.STRIDE_SWAP_THRESHOLD:
            side = -1
        else:
            side = 0  # in the deadzone - too ambiguous to count either way

        if side == 0 or side == self._lead_side:
            # Neutral zone, or still on the current lead side - nothing swap-worthy is
            # in progress.
            self._pending_side = 0
            self._pending_sim_time = 0.0
            self._pending_foot_down = False
            return 0.0

        if side == self._pending_side:
            self._pending_sim_time += self._last_sim_dt
        else:
            self._pending_side = side
            self._pending_sim_time = self._last_sim_dt
            self._pending_foot_down = False

        # side=1 means the left thigh is relatively more forward, so it's the left
        # foot we're watching for a footstep from (and the right foot for side=-1).
        # Latched (not re-checked every call) since the foot is only briefly "down"
        # around the plant, not for the whole dwell window - swing phase brings it back
        # up well before the next swap begins.
        foot_key = "leftfootdown" if side == 1 else "rightfootdown"
        if state.get(foot_key, False):
            self._pending_foot_down = True

        footstep_satisfied = self._pending_foot_down or not self.require_footstep
        if self._pending_sim_time < self.STRIDE_MIN_DWELL_SEC or not footstep_satisfied:
            return 0.0

        # The new side has held for long enough (with a real footstep) to count as a
        # real swap. The very first side settled into after reset (_lead_side still 0)
        # establishes the baseline rather than counting/rewarding a "swap" from nothing.
        is_real_swap = self._lead_side != 0
        if is_real_swap:
            self._episode_stride_count += 1
        bonus = self.stride_reward_coef * velx if is_real_swap else 0.0
        self._lead_side = side
        self._pending_side, self._pending_sim_time, self._pending_foot_down = 0, 0.0, False
        return bonus

    def compute_reward(self, state: Dict[str, float] = None) -> float:
        """Compute shaped reward to guide exploration and learning."""
        state = state or self.get_state()

        velx = state.get("velx", 0.0)
        # No dense per-tick velocity reward by default (speed_reward_coef=0) - forward
        # progress pays out at the moment a stride actually completes (see
        # _stride_bonus/stride_reward_coef), scaled by velx^2 at that instant. This ties
        # earning any speed credit at all to genuinely alternating legs: gaits that
        # never swap (e.g. knee-sliding) get zero reward for their speed, no matter how
        # fast they're moving, since there's no swap event to pay it out on. It also
        # closes off farming swaps in place for free reward - a swap with ~0 velx at
        # that instant pays ~0 regardless of how fast the dwell timer lets swaps
        # repeat. The height term is a floored bonus (see posy_reward_coef docstring
        # above), not a penalty, and stays dense every tick - posture and
        # forward-progress credit are separate concerns.
        height_bonus = max(0.0, -state.get("posy", 0.0))
        speed_bonus = velx * abs(velx)
        # win_bonus is naturally one-shot: state["won"] only ever turns True on the
        # terminal tick (the episode ends immediately after), so there's no dwell-time
        # or repeat-prevention logic needed here the way _stride_bonus needs. Inverse-
        # time (not flat, not target-relative) so a faster finish outscores a slower
        # one with no hand-tuned target to maintain - see win_reward_coef docstring.
        won = state.get("won", False)
        win_bonus = 1.0 / self._episode_sim_time if won and self._episode_sim_time > 0 else 0.0
        reward = (
            self.speed_reward_coef * speed_bonus
            + self.posy_reward_coef * height_bonus
            + self.win_reward_coef * win_bonus
            + self._stride_bonus(state, speed_bonus)
        )

        return float(reward)

    def render(self, mode='human', rank=None, state=None):
        """Print human-readable state. Pass `state` (e.g. the info['state'] a just-completed
        step() already returned) to avoid an extra round trip re-fetching it."""
        state = state if state is not None else self.get_state()
        rank_str = f"[Agent {rank}] " if rank is not None else ""
        print(
            f"\r{rank_str}[QWOP] Pos: {state.get('posx', 0):6.2f}  "
            f"Height: {state.get('posy', 0):5.2f}  "
            f"Vel: {state.get('velx', 0):6.2f}  "
            f"Terminal: {state.get('terminal', False)}",
            end="", flush=True
        )

    def _signal_handler(self, sig, frame):
        """Handle system signals."""
        logger.info(f"Received signal {sig}, cleaning up...")
        self.close()
