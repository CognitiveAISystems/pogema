import string
import sys
import warnings
from contextlib import closing

import numpy as np
from gym import utils
from io import StringIO

from pogema.generator import generate_obstacles, generate_positions_and_targets_fast
from .grid_config import GridConfig


class Grid:

    def __init__(self, grid_config: GridConfig, add_artificial_border: bool = True, num_retries=10):

        self.config = grid_config
        self.rnd = np.random.default_rng(grid_config.seed)

        if self.config.map is None:
            obstacles = generate_obstacles(self.config)
            starts_xy, finishes_xy = generate_positions_and_targets_fast(obstacles, self.config)
        else:
            obstacles = np.array([np.array(line) for line in self.config.map])
            starts_xy, finishes_xy = generate_positions_and_targets_fast(obstacles, self.config)

        if len(starts_xy) != len(finishes_xy):
            for attempt in range(num_retries):
                if len(starts_xy) == len(finishes_xy):
                    warnings.warn(f'Created valid configuration only with {attempt} attempts.', Warning, stacklevel=2)
                    break
                if self.config.map is None:
                    obstacles = generate_obstacles(self.config)
                starts_xy, finishes_xy = generate_positions_and_targets_fast(obstacles, self.config)

        if not starts_xy or not finishes_xy or len(starts_xy) != len(finishes_xy):
            raise OverflowError("Can't create task. Please check grid config, especially density, num_agent and map.")

        if add_artificial_border:
            r = self.config.obs_radius
            if grid_config.empty_outside:
                filled_obstacles = np.zeros(np.array(obstacles.shape) + r * 2)
            else:
                filled_obstacles = self.rnd.binomial(1, grid_config.density, np.array(obstacles.shape) + r * 2)

            height, width = filled_obstacles.shape
            filled_obstacles[r - 1, r - 1:width - r + 1] = grid_config.OBSTACLE
            filled_obstacles[r - 1:height - r + 1, r - 1] = grid_config.OBSTACLE
            filled_obstacles[height - r, r - 1:width - r + 1] = grid_config.OBSTACLE
            filled_obstacles[r - 1:height - r + 1, width - r] = grid_config.OBSTACLE
            filled_obstacles[r:height - r, r:width - r] = obstacles

            obstacles = filled_obstacles

            starts_xy = [(x + r, y + r) for x, y in starts_xy]
            finishes_xy = [(x + r, y + r) for x, y in finishes_xy]

        filled_positions = np.zeros(obstacles.shape)
        for x, y in starts_xy:
            filled_positions[x, y] = 1

        self.obstacles = obstacles
        self.positions = filled_positions
        self.finishes_xy = finishes_xy
        self.positions_xy = starts_xy
        self.hidden_agents = set()

    def get_observation_shape(self):
        full_radius = self.config.obs_radius * 2 + 1
        return 2, full_radius, full_radius

    def get_num_actions(self):
        return len(self.config.MOVES)

    def get_obstacles(self, agent_id):
        x, y = self.positions_xy[agent_id]
        r = self.config.obs_radius
        return self.obstacles[x - r:x + r + 1, y - r:y + r + 1]

    def get_positions(self, agent_id):
        x, y = self.positions_xy[agent_id]
        r = self.config.obs_radius
        return self.positions[x - r:x + r + 1, y - r:y + r + 1]

    def get_target(self, agent_id):

        x, y = self.positions_xy[agent_id]
        fx, fy = self.finishes_xy[agent_id]
        if x == fx and y == fy:
            return 0.0, 0.0
        rx, ry = fx - x, fy - y
        dist = np.sqrt(rx ** 2 + ry ** 2)
        return rx / dist, ry / dist

    def get_square_target(self, agent_id):
        c = self.config
        full_size = self.config.obs_radius * 2 + 1
        result = np.zeros((full_size, full_size))
        x, y = self.positions_xy[agent_id]
        fx, fy = self.finishes_xy[agent_id]
        dx, dy = x - fx, y - fy

        dx = min(dx, c.obs_radius) if dx >= 0 else max(dx, -c.obs_radius)
        dy = min(dy, c.obs_radius) if dy >= 0 else max(dy, -c.obs_radius)
        result[c.obs_radius - dx, c.obs_radius - dy] = 1
        return result

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        chars = string.digits + string.ascii_letters + string.punctuation
        positions_map = {(x, y): id_ for id_, (x, y) in enumerate(self.positions_xy)}
        finishes_map = {(x, y): id_ for id_, (x, y) in enumerate(self.finishes_xy)}
        for line_index, line in enumerate(self.obstacles):
            out = ''
            for cell_index, cell in enumerate(line):
                if cell == self.config.FREE:
                    agent_id = positions_map.get((line_index, cell_index), None)
                    finish_id = finishes_map.get((line_index, cell_index), None)

                    if agent_id is not None:
                        out += str(utils.colorize(' ' + chars[agent_id % len(chars)] + ' ', color='red', bold=True,
                                                  highlight=False))
                    elif finish_id is not None:
                        out += str(
                            utils.colorize('|' + chars[finish_id % len(chars)] + '|', 'white', highlight=False))
                    else:
                        out += str(utils.colorize(str(' . '), 'white', highlight=False))
                else:
                    out += str(utils.colorize(str('   '), 'cyan', bold=False, highlight=True))
            out += '\n'
            outfile.write(out)

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def move(self, agent_id, action):
        x, y = self.positions_xy[agent_id]

        self.positions[x, y] = self.config.FREE

        dx, dy = self.config.MOVES[action]
        if self.obstacles[x + dx, y + dy] == self.config.FREE and self.positions[x + dx, y + dy] == self.config.FREE:
            x += dx
            y += dy

        self.positions_xy[agent_id] = (x, y)
        self.positions[x, y] = self.config.OBSTACLE

    def on_goal(self, agent_id):
        return self.positions_xy[agent_id] == self.finishes_xy[agent_id]

    def hide_agent(self, agent_id):
        if agent_id in self.hidden_agents:
            return False

        self.hidden_agents.add(agent_id)

        x, y = self.positions_xy[agent_id]
        self.positions[x, y] = self.config.FREE

        return True


class CooperativeGrid(Grid):
    def move(self, agent_id, action):
        x, y = self.positions_xy[agent_id]
        fx, fy = self.finishes_xy[agent_id]
        dx, dy = self.config.MOVES[action]
        if self.obstacles[x + dx, y + dy] == self.config.FREE:
            if self.positions[x + dx, y + dy] == self.config.FREE:
                self.positions[x, y] = self.config.FREE
                x += dx
                y += dy
                self.positions[x, y] = self.config.OBSTACLE
        self.positions_xy[agent_id] = (x, y)
        return x == fx and y == fy
