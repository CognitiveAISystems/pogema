from typing import Union

from pogema import GridConfig
from pogema.envs import Pogema
from pogema.wrappers.global_state_info import GlobalStateInfo
from pogema.wrappers.metrics import MetricsWrapper
from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.wrappers.pymarl_integration import PyMarlPogema
from pogema.wrappers.sf_integration import AutoResetWrapper, IsMultiAgentWrapper


def _make_pogema(grid_config):
    env = Pogema(config=grid_config)  # all
    env = GlobalStateInfo(env)  # all
    env = MultiTimeLimit(env, grid_config.max_episode_steps)  # all
    env = MetricsWrapper(env)  # all

    return env


def _make_sample_factory_integration(grid_config):
    env = Pogema(config=grid_config)  # all
    env = GlobalStateInfo(env)  # all
    env = MultiTimeLimit(env, grid_config.max_episode_steps)  # all

    env = MetricsWrapper(env, group_name='episode_extra_stats')  # group name for sample factory
    env = IsMultiAgentWrapper(env)
    env = AutoResetWrapper(env)  # sample factory

    return env


def _make_py_marl_integration(grid_config, *args, **kwargs):
    env = _make_pogema(grid_config)
    return PyMarlPogema(env, grid_config)


def _make_sb3_integration(grid_config):
    raise NotImplementedError


def _make_rllib_integration(grid_config):
    raise NotImplementedError


def _make_single_agent_gym(grid_config):
    raise NotImplementedError


def make_pogema(grid_config: Union[GridConfig, dict] = GridConfig(), *args, **kwargs):
    if isinstance(grid_config, dict):
        grid_config = GridConfig(**grid_config)

    if grid_config.integrations is None:
        return _make_pogema(grid_config)
    elif grid_config.integrations == 'SampleFactory':
        return _make_sample_factory_integration(grid_config)
    elif grid_config.integrations == 'PyMARL':
        return _make_py_marl_integration(grid_config, *args, **kwargs)
    elif grid_config.integrations == 'StableBaselines3':
        return _make_sb3_integration(grid_config)
    elif grid_config.integrations == 'rllib':
        return _make_rllib_integration(grid_config)
    elif grid_config.integrations == 'single_agent_gym':
        assert grid_config.num_agents == 1
        return _make_single_agent_gym
    else:
        raise KeyError(grid_config.integrations)


def main():
    env = _make_sample_factory_integration(GridConfig(num_agents=10))
    env.reset()
    print(env.num_agents)


if __name__ == '__main__':
    main()
