# gym-qwop

A [Gymnasium](https://gymnasium.farama.org/) environment for [QWOP](http://www.foddy.net/Athletics.html),
Bennett Foddy's notoriously difficult sprinting game. The agent controls the Q/W/O/P
keys to run a real 100 meter dash against the game's actual, unmodified physics
engine, driven headlessly through Selenium.

Originally built as part of [RL-QWOP](https://github.com/drakesvoboda/RL-QWOP), a
project to train an RL policy to beat QWOP, and later extracted into its own
installable package. The game itself is a patched build forked from
[juanto121/qwop-ai](https://github.com/juanto121/qwop-ai).

## Installation

Requires [Node.js](https://nodejs.org/) and Firefox (driven via
[Selenium](https://www.selenium.dev/)) to actually run the game. The Node server and
patched game files ship inside the package - nothing else to set up.

```bash
pip install gym-qwop
```

## Usage

```python
import gymnasium as gym
import gym_qwop  # registers qwop-v0, qwop-realtime-v0, qwop-authentic-v0

env = gym.make("qwop-v0")
obs, info = env.reset()

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## Environments

Three variants trade off determinism against realism differently:

* **`qwop-v0`** - lock-step. Physics only advances when explicitly driven by the
  agent, decoupled from wall-clock time. This is the one to train against: fast,
  deterministic parallel rollouts, with no dependence on real-time scheduling.
* **`qwop-realtime-v0`** - physics runs on its own, independent of the agent, at an
  approximation of real-time pacing (or faster, for training throughput).
* **`qwop-authentic-v0`** - physics is paced by the browser's own native
  `requestAnimationFrame` scheduler, completely untouched. The only mode that's a
  genuinely fair, human-comparable timing benchmark - use this one for any run you
  want to compare against a human time.

Across all three, the underlying Box2D physics solver always integrates at a fixed
25Hz (`world.step(0.04, 5, 5)`) - separate from the original game's own 30Hz render
loop.

## Parameters

All three variants accept `gym.make("qwop-v0", **kwargs)`-style keyword arguments.

**Connection & setup** (shared by all three)

| Param | Default | |
|---|---|---|
| `url` | `None` | Game server URL. Falls back to the `QWOP_URL` env var, then `http://localhost:<server_port>`. |
| `server_port` | `None` | Port for the game server. Falls back to `QWOP_SERVER_PORT`, then `3100`. Each parallel instance needs its own port. |
| `auto_start_server` | `True` | Start the bundled Node server automatically if nothing is listening on `server_port` yet. |
| `server_path` | `None` | Path to a game server to use instead of the one bundled with the package. Falls back to `QWOP_SERVER_PATH`. |
| `timeout` | `10` | Page load timeout (seconds). Falls back to `QWOP_TIMEOUT`. |
| `enable_sound` | `False` | Enable browser audio. |
| `render_fps` | `60` | Only affects `render_mode="human"`'s pacing. |

**Reward shaping** (shared by all three)

| Param | Default | |
|---|---|---|
| `speed_reward_coef` | `0.0` | Weight on the dense per-tick forward-speed term (`velx * abs(velx)`). `0` makes all forward-progress credit gated behind completing a stride instead. |
| `stride_reward_coef` | `1.0` | Weight on a bonus paid the instant the lead leg actually swaps sides - a genuine stride, not just a thigh-angle crossing. |
| `posy_reward_coef` | `0.0` | Weight on a standing-height *bonus* (never a penalty - see the docstring in `qwop_selenium_base.py` for why that distinction matters for the incentives it creates). |
| `win_reward_coef` | `0.0` | Weight on a one-time bonus paid on reaching the 100m finish, scaled as `coef / episode_sim_time` - the only term that directly rewards finishing *fast* rather than just moving well. |
| `require_footstep` | `True` | Require the swapped leg to actually plant on the ground (not just swing past an angle threshold) before a stride counts. |

**Action space** (shared by all three)

| Param | Default | |
|---|---|---|
| `action_mode` | `"categorical"` | `"categorical"`: `Discrete(16)`, one class per key combination - can represent correlations between keys. `"bernoulli"`: `MultiBinary(4)`, 4 independent per-key probabilities instead - no cross-key correlation, but pairs naturally with CAPS-style smoothness regularization since each penalized probability maps to exactly one key. |
| `include_key_state` | `True` | Include the 4 currently-held keys in the observation. Needed for any temporal-smoothness regularizer to have something stable to learn ("keep holding this key" can't be a function of the input without it). |

**Timing** (differs per variant)

| Env | Param | Default | |
|---|---|---|---|
| `qwop-v0` | `ticks_per_step` | `1` | Exact number of physics ticks advanced per `step()` call, driven directly by Python - deterministic lock-step, decoupled from wall-clock time. |
| `qwop-realtime-v0` | `min_ticks_per_step` | `1` | Minimum new physics ticks that must land before `step()` returns, since physics advances on its own independent timer. |
| `qwop-realtime-v0` | `speed_multiplier` | `1.0` | Runs the independent timer faster than true wall-clock (for training throughput) while still keeping it agent-independent. `1.0` is genuine real-time. |
| `qwop-authentic-v0` | `min_ticks_per_step` | `1` | Same floor as above, but physics is paced by the browser's own native scheduler underneath - this one is genuinely real-time, not an approximation of it. |

## Configuration

The game server auto-starts on an available port and shuts down with the environment.
Override where it's found with the `QWOP_SERVER_PATH` environment variable if you need
to run a modified copy instead of the one bundled with the package.

## License

MIT
