from copy import deepcopy

import gym

from pogema import GridConfig
from pogema.envs import Pogema


class GlobalStateInfo(gym.Wrapper):
    def __init__(self, env: Pogema):
        super().__init__(env)

    def get_obstacles(self, ignore_borders=False):
        gc: GridConfig = self.env.config
        if ignore_borders:
            return self.env.grid.obstacles[gc.obs_radius:-gc.obs_radius, gc.obs_radius:-gc.obs_radius].copy()
        return self.env.grid.obstacles.copy()

    @staticmethod
    def _cut_borders_xy(positions, obs_radius):
        return [[x - obs_radius, y - obs_radius] for x, y in positions]

    @staticmethod
    def _filter_inactive(pos, active_flags):
        return [pos for idx, pos in enumerate(pos) if active_flags[idx]]

    def _get_grid_config(self) -> GridConfig:
        return self.env.config

    def _prepare_positions(self, positions, only_active, ignore_borders):
        gc = self._get_grid_config()

        if only_active:
            positions = self._filter_inactive(positions, self.env.active)

        if ignore_borders:
            positions = self._cut_borders_xy(positions, gc.obs_radius)

        return positions

    def get_agents_xy(self, only_active=False, ignore_borders=False):
        return self._prepare_positions(deepcopy(self.env.grid.positions_xy), only_active, ignore_borders)

    def get_targets_xy(self, only_active=False, ignore_borders=False):
        return self._prepare_positions(deepcopy(self.env.grid.finishes_xy), only_active, ignore_borders)


def main():
    pogema_map = """
    a.#.#.
    A.#.#.
    #.#.#.
    .#.#.#
    .#.#z.
    .#.#..
    .#.#.#
    #.#.#.
    #.#Z#.
    #.#.#.
    """
    env = Pogema(config=GridConfig(map=pogema_map, obs_radius=3))
    env = GlobalStateInfo(env)
    env.reset()

    # print(env.get_targets_xy())
    obstacles = env.get_obstacles(ignore_borders=True)
    x, y = env.get_agents_xy(ignore_borders=True)[0]
    tx, ty = env.get_targets_xy(ignore_borders=True)[0]
    obstacles[x, y] = 2.0
    obstacles[tx, ty] = 3.0
    print(obstacles)

    for _ in range(100):
        env.step([env.action_space.sample() for _ in range(env.config.num_agents)])
        print(env.get_agents_xy(ignore_borders=True, only_active=True),
              env.get_targets_xy(ignore_borders=True, only_active=True))


if __name__ == '__main__':
    main()
