from typing import Union

import gym

from pogema import GridConfig
from pogema.envs import Pogema
from pogema.wrappers.global_state_info import GlobalStateInfo
from pogema.wrappers.metrics import MetricsWrapper
from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.wrappers.pymarl_integration import PyMarlPogema
from pogema.wrappers.sf_integration import AutoResetWrapper, IsMultiAgentWrapper


def _make_pogema(grid_config):
    env = Pogema(config=grid_config)
    env = GlobalStateInfo(env)
    env = MultiTimeLimit(env, grid_config.max_episode_steps)
    env = MetricsWrapper(env)

    return env


def _make_sample_factory_integration(grid_config):
    env = _make_pogema(grid_config)
    env.update_group_name(group_name='episode_extra_stats')
    env = IsMultiAgentWrapper(env)
    env = AutoResetWrapper(env)  # sample factory

    return env


def _make_py_marl_integration(grid_config, *_, **__):
    return PyMarlPogema(_make_pogema(grid_config), grid_config)


def _make_sb3_integration(grid_config):
    raise NotImplementedError


def _make_rllib_integration(grid_config):
    raise NotImplementedError


class SingleAgentWrapper(gym.Wrapper):

    def step(self, action):
        observations, rewards, dones, infos = self.env.step([action])
        return observations[0], rewards[0], dones[0], infos[0]

    def reset(self):
        return self.env.reset()[0]


def _make_single_agent_gym(grid_config):
    env = _make_pogema(grid_config)
    env = SingleAgentWrapper(env)

    return env


def make_pogema(grid_config: Union[GridConfig, dict] = GridConfig(), *args, **kwargs):
    if isinstance(grid_config, dict):
        grid_config = GridConfig(**grid_config)

    if grid_config.integration is None:
        return _make_pogema(grid_config)
    elif grid_config.integration == 'SampleFactory':
        return _make_sample_factory_integration(grid_config)
    elif grid_config.integration == 'PyMARL':
        return _make_py_marl_integration(grid_config, *args, **kwargs)
    elif grid_config.integration == 'StableBaselines3':
        return _make_sb3_integration(grid_config)
    elif grid_config.integration == 'rllib':
        return _make_rllib_integration(grid_config)
    elif grid_config.integration == 'single_agent_gym':
        assert grid_config.num_agents == 1
        return _make_single_agent_gym(grid_config)

    raise KeyError(grid_config.integration)


def main():
    pass


if __name__ == '__main__':
    main()
