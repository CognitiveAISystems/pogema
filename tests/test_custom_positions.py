import numpy as np
from pydantic import ValidationError
import gym
from pogema import GridConfig
from pogema.grid import Grid
import pytest

def test_custom_starts_and_finishes_map():

    grid = """
    .....#.....
    .....#.....
    ...........
    .....#.....
    .....#.....
    #.####.....
    .....###.##
    .....#.....
    .....#.....
    ...........
    .....#.....
    """
    grid_config = GridConfig(map=grid, num_agents=2, agents_xy=[(0,0), (1,1)], targets_xy=[(2,2), (3,3)])
    env = gym.make('Pogema-v0', grid_config=grid_config)
    obs = env.reset()
    r = grid_config.obs_radius
    assert [(x - r, y - r) for x, y in env.grid.positions_xy] == [(0,0), (1,1)] and \
        [(x - r, y - r) for x, y in env.grid.finishes_xy] == [(2,2), (3,3)]

def test_custom_starts_and_finishes_random():
    agents_xy = [(x,x) for x in range(8)]
    targets_xy = [(x,x) for x in range(8,16)]
    grid_config = GridConfig(size=16, num_agents=8, agents_xy=agents_xy, targets_xy=targets_xy)
    env = gym.make('Pogema-v0', grid_config=grid_config)
    obs = env.reset()
    r = grid_config.obs_radius
    assert [(x - r, y - r) for x, y in env.grid.positions_xy] == agents_xy and \
        [(x - r, y - r) for x, y in env.grid.finishes_xy] == targets_xy