import numpy as np

from pogema.grid import GridConfig
from pogema.envs import Pogema


class ActionMapping:
    noop: int = 0
    up: int = 1
    down: int = 2
    left: int = 3
    right: int = 4


def test_moving():
    env = Pogema(config=GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42))
    ac = ActionMapping()
    env.reset()

    env.step([ac.right, ac.noop])
    env.step([ac.up, ac.noop])
    env.step([ac.left, ac.noop])
    env.step([ac.down, ac.noop])
    env.step([ac.down, ac.noop])
    env.step([ac.left, ac.noop])
    env.step([ac.left, ac.noop])
    env.step([ac.up, ac.noop])
    env.step([ac.up, ac.noop])
    env.step([ac.up, ac.noop])

    env.step([ac.right, ac.noop])
    env.step([ac.up, ac.noop])
    env.step([ac.right, ac.noop])
    env.step([ac.down, ac.noop])
    obs, reward, done, infos = env.step([ac.right, ac.noop])

    assert np.isclose([1.0, 0.0], reward).all()
    assert np.isclose([True, False], done).all()


def test_types():
    env = Pogema(config=GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42))
    obs = env.reset()

    # todo replace float64 with float32 in grid and add tests
    # print(obs[0].dtype)
