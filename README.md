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

## Configuration

The game server auto-starts on an available port and shuts down with the environment.
Override where it's found with the `QWOP_SERVER_PATH` environment variable if you need
to run a modified copy instead of the one bundled with the package.

## License

MIT
