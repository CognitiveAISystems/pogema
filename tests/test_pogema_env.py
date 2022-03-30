import numpy as np

from pogema.grid import GridConfig
from pogema.integrations.make_pogema import make_pogema


class ActionMapping:
    noop: int = 0
    up: int = 1
    down: int = 2
    left: int = 3
    right: int = 4


def test_moving():
    env = make_pogema(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42))
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
    env = make_pogema(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42))
    obs = env.reset()

    # todo replace float64 with float32 in grid and add tests
    # print(obs[0].dtype)


def run_episode(grid_config):
    env = make_pogema(grid_config)
    env.reset()

    obs, rewards, dones, infos = env.reset(), [None], [False], [None]

    results = [[obs, rewards, dones, infos]]
    while not all(dones):
        results.append(env.step(env.sample_actions()))
        dones = results[-1][-2]
    return results


def test_metrics():
    _, _, _, infos = run_episode(GridConfig(num_agents=2, seed=5, size=5, max_episode_steps=64))[-1]
    assert np.isclose(infos[0]['metrics']['CSR'], 0.0)
    assert np.isclose(infos[0]['metrics']['ISR'], 0.0)
    assert np.isclose(infos[1]['metrics']['ISR'], 1.0)

    _, _, _, infos = run_episode(GridConfig(num_agents=2, seed=5, size=5, max_episode_steps=512))[-1]
    assert np.isclose(infos[0]['metrics']['CSR'], 1.0)
    assert np.isclose(infos[0]['metrics']['ISR'], 1.0)
    assert np.isclose(infos[1]['metrics']['ISR'], 1.0)

    _, _, _, infos = run_episode(GridConfig(num_agents=5, seed=5, size=5, max_episode_steps=64))[-1]
    assert np.isclose(infos[0]['metrics']['CSR'], 0.0)
    assert np.isclose(infos[0]['metrics']['ISR'], 0.0)
    assert np.isclose(infos[1]['metrics']['ISR'], 0.0)
    assert np.isclose(infos[2]['metrics']['ISR'], 0.0)
    assert np.isclose(infos[3]['metrics']['ISR'], 1.0)
    assert np.isclose(infos[4]['metrics']['ISR'], 0.0)
