from copy import deepcopy

import gym

from pogema import GridConfig
import numpy as np


class GlobalStateInfo(gym.Wrapper):
    def __init__(self, env):
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

    def get_grid_config(self):
        return deepcopy(self._get_grid_config())

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

    def _normalize_coordinates(self, coordinates):
        gc = self._get_grid_config()

        x, y = coordinates

        x -= gc.obs_radius
        y -= gc.obs_radius

        x /= gc.size - 1
        y /= gc.size - 1

        return x, y

    def get_state(self, ignore_borders=False, as_dict=False):
        agents_xy = list(map(self._normalize_coordinates, self.get_agents_xy(ignore_borders)))
        targets_xy = list(map(self._normalize_coordinates, self.get_targets_xy(ignore_borders)))

        obstacles = self.get_obstacles(ignore_borders)

        if as_dict:
            return {"obstacles": obstacles, "agents_xy": agents_xy, "targets_xy": targets_xy}

        return np.concatenate(list(map(lambda x: np.array(x).flatten(), [agents_xy, targets_xy, obstacles])))
