from gym import register
from pogema.grid_config import GridConfig

__version__ = '1.0.3'

__all__ = [
    'GridConfig',
]

register(
    id="Pogema-v0",
    entry_point="pogema.integrations.make_pogema:make_pogema",
)


def _get_num_agents_by_target_density(_size, _agent_density, _obstacle_density):
    return round(_agent_density * (_size * _size * (1.0 - _obstacle_density)))


for size, max_episode_steps in zip([8, 16, 32, 64], [64, 128, 256, 512]):
    for obstacle_density in [0.3]:
        for difficulty, agent_density in zip(['easy', 'normal', 'hard', 'extra-hard'],
                                             [0.0223, 0.0446, 0.0892, 0.1784]):
            num_agents = _get_num_agents_by_target_density(size, agent_density, obstacle_density)
            register(
                id=f'Pogema-{size}x{size}-{difficulty}-v0',
                entry_point="pogema.integrations.make_pogema:make_pogema",
                kwargs={"grid_config": GridConfig(size=size, num_agents=num_agents, density=obstacle_density,
                                                  max_episode_steps=max_episode_steps),
                        "integration": None})
