"""Base environment interface for QWOP.

Also holds the infrastructure shared by every Selenium-driven environment variant
(qwop_env.py, qwop_realtime_env.py, qwop_authentic_env.py): starting the Node game
server, launching a Firefox tab to run it, frame-rate throttling, and resource cleanup.
"""
import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import numpy as np
import gymnasium
from gymnasium import spaces
from selenium import webdriver
from selenium.common.exceptions import WebDriverException

from .server import GameServerManager

logger = logging.getLogger(__name__)


class BaseQWOPEnv(gymnasium.Env, ABC):
    """Abstract base class for QWOP environments."""

    metadata = {'render_modes': ['human', 'realtime'], 'render_fps': 60}

    # The hurdle's resting x-position (QWOP.min.js's create_hurdle: hurdleAt=1e4 world
    # units / worldScale=20 / the game's *10 posx scaling = 50, matching state.posx's
    # units) - used only as a fallback default if state["hurdletop_x"] is ever missing.
    # The hurdle is NOT static once play reaches it: create_hurdle builds hurdleTop and
    # hurdleBase as two separate dynamic Box2D bodies joined by a B2RevoluteJointDef
    # (a hinge, enableLimit=true - it can tip/fall within a bounded rotation range on
    # contact, not just sit there). A fixed HURDLE_POSX - posx distance goes stale the
    # instant the hurdle actually gets hit, i.e. exactly when its state matters most -
    # obs() below reads the live hurdletop_x/hurdletop_angle QWOP.min.js now reports
    # per-tick (same place torso/head angles etc. are read) instead.
    HURDLE_POSX = 50.0

    # Absolute index of the ticks_elapsed dimension within obs() - always here
    # regardless of include_key_state, since key-state dims are appended after it (see
    # obs() below). Used by rlqwop.ActorCritic's obs_override backward-compat shim for
    # checkpoints trained under a fixed ticks_per_step (see that class's docstring).
    TICKS_ELAPSED_OBS_INDEX = 26

    def __init__(self, render_mode=None, render_fps=60, action_mode="categorical",
                 include_key_state=True):
        """Initialize observation and action spaces.

        Args:
            action_mode: 'categorical' (default) - Discrete(16), one class per
                combination of the 4 keys (Q, W, O, P) being up/down, modeled as a single
                Categorical rather than 4 independent Bernoullis so the policy can
                represent correlations between keys (e.g. always pressing two specific
                keys together) that independent per-key probabilities structurally can't
                express. 'bernoulli' - MultiBinary(4), 4 independent per-key Bernoullis
                instead; the action *is* the [q, w, o, p] array directly, no decoding by
                bit position needed. The two matter for CAPS smoothness regularization
                (see run_update's caps_terms in rlqwop.PPO): under 'categorical', the 16
                class indices are an arbitrary bit-packing with no relationship to how
                many keys actually flip between two actions, so smoothing the
                distribution doesn't necessarily smooth the keypresses; under
                'bernoulli', each of the 4 output probabilities maps directly to one
                key's likelihood, so the same regularizer penalizes actual key-flip rate.
            include_key_state: If True (default), append the 4 currently-held keys to
                the observation (see obs()) - lets the policy represent "keep holding
                this key" as a stable function of its input, which matters for CAPS's
                temporal smoothness term (without it, the same body pose can call for
                different actions depending on how long a key's already been held, since
                gait phase isn't recoverable from instantaneous body state alone). Set to
                False to omit it (28-dim observation instead of 32).
        """
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.include_key_state = include_key_state

        if action_mode == "categorical":
            self.action_space = spaces.Discrete(16)
        elif action_mode == "bernoulli":
            self.action_space = spaces.MultiBinary(4)
        else:
            raise ValueError(f"Unknown action_mode: {action_mode!r} (expected 'categorical' or 'bernoulli')")
        self.action_mode = action_mode

        # State vector: [posy, then (sin(angle), cos(angle)) for each of the 8 limb
        # joints, 8 joint velocities, velx, ticks_elapsed, hurdle_dx, (sin(angle),
        # cos(angle)) for the hurdle's own tilt] plus, if include_key_state, [q, w, o, p]
        # (currently-held key state, bit order matching _decode_action). The hurdle
        # angle dims were appended after hurdle_dx (not inserted earlier) specifically
        # so every existing index up through hurdle_dx - including
        # TICKS_ELAPSED_OBS_INDEX - stays valid; only key_state's indices (if present)
        # shift by 2.
        #
        # Angles are sin/cos-encoded rather than raw radians because Box2D's revolute
        # joints aren't wrapped to [-pi, pi] - a limb can spin past a full rotation under
        # continuous torque (see the wrap_to_pi stride-detection comment in
        # qwop_selenium_base.py) - so the same physical pose can land arbitrarily far
        # apart in the raw scalar (e.g. 3.0 rad vs 3.0 + 2*pi rad after one extra spin).
        # That discontinuity breaks both normalize_obs's running mean/var (a physically
        # tiny change producing a huge raw delta) and CAPS's "nearby observations map to
        # nearby actions" assumption (see obs() and rlqwop.PPO's caps_terms) right at the
        # wrap boundary. sin/cos is periodic and bounded to [-1, 1] by construction, so
        # identical physical poses always encode identically regardless of accumulated
        # rotations, and small angle changes always produce small encoding changes.
        #
        # Total: 1 + 8*2 + 8 + 1 + 1 + 1 + 2 (+ 4) = 30 or 34 dimensions.
        obs_dim = 30 + (4 if include_key_state else 0)
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        low = -high
        # The 16 joint sin/cos dims (right after posy) and the 2 hurdle-tilt sin/cos
        # dims (right after hurdle_dx, at the very end before key_state) are bounded to
        # [-1, 1] by construction, unlike every other dimension here.
        low[1:17] = -1.0
        high[1:17] = 1.0
        low[28:30] = -1.0
        high[28:30] = 1.0
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.last_reward = 0.0
        self.last_state_time = 0.0
        self.old_posx = 0.0
        self.old_velx = 0.0
        # Keys currently held down in the browser, as of the last obs() call - set by each
        # subclass's step()/reset() from the action it just applied (or all-zero right
        # after reset, which releases every key), regardless of include_key_state (cheap
        # to track; obs() is what decides whether it's actually exposed to the policy).
        self._current_keys = np.zeros(4, dtype=np.float32)

        self.driver = None
        self.server_manager = None
        self.max_fps = render_fps
        self.last_frame_time = time.time()

    def _resolve_server_port(self, default: int) -> int:
        """Resolve this instance's game server port: explicit arg < QWOP_SERVER_PORT env
        var < `default`. Each parallel training worker needs a distinct port (rlqwop.PPO
        sets QWOP_SERVER_PORT per worker when running with num_procs > 1)."""
        return int(os.getenv("QWOP_SERVER_PORT", str(default)))

    def _start_game_server(self, server_path: Optional[str], server_port: int, timeout: int) -> Optional[GameServerManager]:
        """Start (or attach to) the Node game server, or None if it couldn't be started."""
        try:
            return GameServerManager(
                server_path=server_path,
                port=server_port,
                auto_start=True,
                timeout=timeout,
            )
        except RuntimeError as e:
            logger.warning(f"Could not start game server: {e}")
            return None

    def _launch_browser(self, url: str, headless: bool, enable_sound: bool = False):
        """Launch a Firefox tab pointed at `url` and return the WebDriver."""
        try:
            options = webdriver.FirefoxOptions()
            if headless:
                options.add_argument("--headless")
            if not enable_sound:
                options.set_preference("media.volume_scale", "0.0")
            # Disable Firefox's background auto-updater entirely. Without this, a
            # freshly auto-updated Firefox binary can pop a macOS system dialog on
            # first launch ("Firefox is trying to install a new helper tool") asking
            # for admin authorization - a GUI prompt that just hangs any Selenium
            # session that happens to trigger it, since nothing here can click through
            # it. Automated/headless instances should never attempt to self-update.
            options.set_preference("app.update.auto", False)
            options.set_preference("app.update.enabled", False)
            options.set_preference("app.update.background.enabled", False)
            # The actual trigger: Selenium gives every launch a brand-new temp profile
            # (rust_mozprofileXXXXX), so Firefox's "have I already run first-run/
            # post-update housekeeping in this profile" check never finds a marker -
            # every single automated launch looks like the first run after whatever
            # version last auto-updated, re-running the post-update flow (which is what
            # actually pops the helper-tool dialog) forever, not just once.
            options.set_preference("app.update.postupdate", False)
            # The four prefs above only stop Firefox from *checking for/applying* new
            # updates going forward - they don't retroactively clear an update that a
            # previous, non-Selenium launch of this same Firefox.app (update staging
            # happens at the app-bundle level, shared across every profile including
            # Selenium's fresh temp ones) already found and staged. On the next launch
            # of ANY profile, a staged update can still trigger the "installing a
            # helper tool" authorization dialog these prefs don't cover, and since
            # nothing here can click through it, Firefox can end up exiting outright
            # (WebDriverException: "Process unexpectedly closed with status 0") rather
            # than hanging. Blanking the update-check URL prevents Firefox from ever
            # successfully finding/staging an update in the first place - the standard
            # fix in CI/enterprise Firefox deployments for this class of issue.
            options.set_preference("app.update.url", "")

            driver = webdriver.Firefox(options=options)
            driver.get(url)
            return driver

        except WebDriverException as e:
            logger.error(f"Failed to launch browser: {e}")
            raise RuntimeError(
                "Failed to start Firefox. Make sure Firefox is installed.\n"
                "Download: https://www.mozilla.org/en-US/firefox/new/\n"
                f"Error: {e}"
            )

    def _throttle(self, target_dt: Optional[float] = None):
        """Sleep as needed to pace stepping to `target_dt` seconds per call (default
        1/self.max_fps) - only when actually rendering ("human" or "realtime" mode).
        Training (render_mode=None) doesn't need real-time pacing, and capping it would
        limit rollout throughput regardless of how fast the sim itself can run.

        render_mode="human" always uses the default 1/max_fps regardless of `target_dt` -
        it's for watching at a fun, fixed frame rate, not literal real-time (see
        render_mode="realtime" for that, which passes the actual amount of simulated time
        each step covers, e.g. ticks_per_step * PHYSICS_DT on QWOPEnv)."""
        if self.render_mode != "realtime":
            return
        if target_dt is None:
            target_dt = 1.0 / self.max_fps
        current_time = time.time()
        delta = current_time - self.last_frame_time
        if delta < target_dt:
            time.sleep(target_dt - delta)
        self.last_frame_time = time.time()

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Returns:
            obs: observation
            reward: reward
            terminated: episode finished (e.g., fell over)
            truncated: episode timeout (max steps reached)
            info: additional info dict
        """
        pass

    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            obs: initial observation
            info: additional info dict
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, float]:
        """Get current state from the environment."""
        pass

    @abstractmethod
    def compute_reward(self, state: Dict[str, float] = None) -> float:
        """Compute reward based on state."""
        pass

    def obs(self, state: Dict[str, float] = None) -> np.ndarray:
        """Convert state to observation vector (posy + sin/cos-encoded joint angles +
        velocities + ticks_elapsed + hurdle_dx + sin/cos-encoded hurdle tilt
        [+ key state])."""
        state = state or self.get_state()
        joint_names = ["head", "torso", "leftarm", "rightarm", "leftthigh", "rightthigh", "leftcalf", "rightcalf"]

        # Positions: posy is a plain height, but each limb angle is encoded as
        # (sin(angle), cos(angle)) rather than the raw radian - see observation_space's
        # comment in __init__ for why (unwrapped joint rotation, wraparound discontinuity).
        obs = [state.get("posy", 0)]
        for joint in joint_names:
            angle = state.get(joint, 0)
            obs.append(np.sin(angle))
            obs.append(np.cos(angle))

        # Velocities (from state if available)
        obs.extend([state.get(f"{joint}_vel", 0) for joint in joint_names])

        # Horizontal velocity
        obs.append(state.get("velx", 0))

        # How many physics ticks the environment just covered to reach this state - see
        # qwop_selenium_base.py's _compute_state. Constant under lock-step; genuinely
        # variable (and otherwise a hidden variable) under the real-time env.
        obs.append(state.get("ticks_elapsed", 0))

        # Signed distance to the hurdle's *current* position (positive = still ahead,
        # negative = already passed) - see HURDLE_POSX above for why this reads the live
        # hurdletop_x QWOP.min.js reports rather than a fixed constant.
        obs.append(state.get("hurdletop_x", self.HURDLE_POSX) - state.get("posx", 0))

        # Hurdle tilt, sin/cos-encoded for the same unwrapped-rotation reason as the
        # limb joint angles above - 0 (sin=0, cos=1) is upright/undisturbed; a hit
        # rotates it within the hinge joint's limit, moving this away from (0, 1) as it
        # tips, all the way to "fallen over" at the limit.
        hurdle_angle = state.get("hurdletop_angle", 0)
        obs.append(np.sin(hurdle_angle))
        obs.append(np.cos(hurdle_angle))

        # Currently-held key state - see include_key_state in __init__.
        if self.include_key_state:
            obs.extend(self._current_keys.tolist())

        return np.array(obs, dtype=np.float32)

    def _decode_action(self, action) -> list:
        """Decode the policy's raw action into the [q, w, o, p] key-state list the game
        expects - see action_mode in __init__. 'bernoulli' actions are already that list
        (as a length-4 array of 0/1-ish values); 'categorical' actions are a Discrete(16)
        index, decoded bit by bit (bit i is key i's state)."""
        if self.action_mode == "bernoulli":
            return [int(round(float(x))) for x in np.asarray(action).flatten()]
        action = int(action)
        return [(action >> i) & 1 for i in range(4)]

    def render(self, mode='human') -> None:
        """Render environment (optional)."""
        pass

    def get_render_state(self) -> str:
        """Get a human-readable representation of current state."""
        state = self.get_state()
        return (
            f"Position: {state.get('posx', 0):.2f}  "
            f"Height: {state.get('posy', 0):.2f}  "
            f"Velocity: {state.get('velx', 0):.2f}  "
            f"Terminal: {state.get('terminal', False)}"
        )

    def close(self) -> None:
        """Clean up the browser and game server. Subclasses with extra resources (e.g. a
        websocket) should close those first, then call super().close()."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Browser closed successfully")
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
            finally:
                self.driver = None

        if self.server_manager:
            try:
                self.server_manager.stop()
            except Exception as e:
                logger.warning(f"Error stopping server: {e}")
            finally:
                self.server_manager = None
