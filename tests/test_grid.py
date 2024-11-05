import numpy as np
from pydantic import ValidationError

from pogema import GridConfig
from pogema.grid import Grid
import pytest

from pogema.integrations.make_pogema import pogema_v0


def test_obstacle_creation():
    config = GridConfig(seed=1, obs_radius=2, size=5, num_agents=1, density=0.2)
    obstacles = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    assert np.isclose(Grid(config).obstacles, obstacles).all()

    config = GridConfig(seed=3, obs_radius=1, size=4, num_agents=1, density=0.4)
    obstacles = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                 [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                 [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                 [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    assert np.isclose(Grid(config).obstacles, obstacles).all()


def test_initial_positions():
    config = GridConfig(seed=1, obs_radius=2, size=5, num_agents=1, density=0.2)
    positions_xy = [(2, 4)]
    assert np.isclose(Grid(config).positions_xy, positions_xy).all()

    config = GridConfig(seed=1, obs_radius=2, size=12, num_agents=10, density=0.2)
    positions_xy = [(13, 10), (7, 4), (4, 3), (2, 11), (12, 6), (8, 11), (6, 8), (2, 12), (2, 10), (9, 11)]
    assert np.isclose(Grid(config).positions_xy, positions_xy).all()


def test_goals():
    config = GridConfig(seed=1, obs_radius=2, size=5, num_agents=1, density=0.4)
    finishes_xy = [(5, 2)]
    assert np.isclose(Grid(config).finishes_xy, finishes_xy).all()

    config = GridConfig(seed=2, obs_radius=2, size=12, num_agents=10, density=0.2)
    finishes_xy = [(11, 10), (8, 11), (2, 13), (3, 5), (12, 6), (9, 12), (9, 6), (9, 2), (10, 2), (6, 11)]
    assert np.isclose(Grid(config).finishes_xy, finishes_xy).all()


def test_overflow():
    with pytest.raises(OverflowError):
        Grid(GridConfig(seed=1, obs_radius=2, size=4, num_agents=100, density=0.0))

    with pytest.raises(OverflowError):
        Grid(GridConfig(seed=1, obs_radius=2, size=4, num_agents=1, density=1.0))


def test_overflow_warning():
    with pytest.warns(Warning):
        for _ in range(1000):
            Grid(GridConfig(obs_radius=2, size=4, num_agents=6, density=0.3), num_retries=10000)


def test_edge_cases():
    with pytest.raises(ValidationError):
        GridConfig(seed=1, obs_radius=2, size=1, num_agents=1, density=0.4)

    with pytest.raises(ValidationError):
        GridConfig(seed=1, obs_radius=2, size=4, num_agents=0, density=0.4)

    with pytest.raises(OverflowError):
        Grid(GridConfig(seed=1, obs_radius=2, size=4, num_agents=1, density=1.0))

    with pytest.raises(ValidationError):
        Grid(GridConfig(seed=1, obs_radius=2, size=4, num_agents=1, density=2.0))


def test_edge_cases_for_custom_map():
    test_map = [[0, 0, 0]]
    with pytest.raises(OverflowError):
        Grid(GridConfig(seed=1, obs_radius=2, size=4, num_agents=2, map=test_map))
    with pytest.raises(OverflowError):
        Grid(GridConfig(seed=2, obs_radius=2, size=4, num_agents=4, map=test_map))


def test_custom_map():
    test_map = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    grid = Grid(GridConfig(seed=1, obs_radius=2, size=4, num_agents=2, map=test_map))
    obstacles = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    assert np.isclose(grid.obstacles, obstacles).all()

    test_map = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]
    grid = Grid(GridConfig(seed=1, obs_radius=2, size=4, num_agents=2, map=test_map))
    obstacles = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    assert np.isclose(grid.obstacles, obstacles).all()

    test_map = [
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1],
    ]
    grid = Grid(GridConfig(seed=1, obs_radius=2, size=4, num_agents=2, map=test_map))
    obstacles = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    assert np.isclose(grid.obstacles, obstacles).all()


def test_overflow_for_custom_map():
    test_map = [
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1],
    ]
    with pytest.raises(OverflowError):
        Grid(GridConfig(obs_radius=2, size=4, num_agents=5, density=0.3, map=test_map), num_retries=100)


def test_str_custom_map():
    grid_map = """
        .a...#.....
        .....#.....
        ..C.....b..
        .....#.....
        .....#.....
        #.####.....
        .....###.##
        .....#.....
        .c...#.....
        .B.......A.
        .....#.....
    """
    grid = Grid(GridConfig(obs_radius=2, size=4, density=0.3, map=grid_map))
    assert (grid.config.num_agents == 3)
    assert (np.isclose(0.1404958, grid.config.density))
    assert (np.isclose(11, grid.config.size))

    grid_map = """.....#...."""
    grid = Grid(GridConfig(seed=2, num_agents=3, map=grid_map))
    assert (grid.config.num_agents == 3)
    assert (np.isclose(0.1, grid.config.density))
    assert (np.isclose(10, grid.config.size))


def test_custom_starts_and_finishes_random():
    agents_xy = [(x, x) for x in range(8)]
    targets_xy = [(x, x) for x in range(8, 16)]
    grid_config = GridConfig(seed=12, size=16, num_agents=8, agents_xy=agents_xy, targets_xy=targets_xy)
    env = pogema_v0(grid_config=grid_config)
    env.reset()
    r = grid_config.obs_radius
    assert [(x - r, y - r) for x, y in env.grid.positions_xy] == agents_xy and \
           [(x - r, y - r) for x, y in env.grid.finishes_xy] == targets_xy


def test_out_of_bounds_for_custom_positions():
    Grid(GridConfig(seed=12, size=17, agents_xy=[[0, 16]], targets_xy=[[16, 0]]))

    with pytest.raises(IndexError):
        GridConfig(seed=12, size=17, agents_xy=[[0, 17]], targets_xy=[[0, 0]])
    with pytest.raises(IndexError):
        GridConfig(seed=12, size=17, agents_xy=[[0, 0]], targets_xy=[[0, 17]])
    with pytest.raises(IndexError):
        GridConfig(seed=12, size=17, agents_xy=[[-1, 0]], targets_xy=[[0, 0]])
    with pytest.raises(IndexError):
        GridConfig(seed=12, size=17, agents_xy=[[0, 0]], targets_xy=[[0, -1]])


def test_duplicated_params():
    grid_map = "Aa"
    with pytest.raises(KeyError):
        GridConfig(agents_xy=[[0, 0]], targets_xy=[[0, 0]], map=grid_map)


def test_custom_grid_with_empty_agents_and_targets():
    grid_map = """...."""
    Grid(GridConfig(agents_xy=None, targets_xy=None, map=grid_map, num_agents=1))


def test_custom_grid_with_specific_positions():
    grid_map = """
        !!!!!!!!!!!!!!!!!!
        !@@!@@!$$$$$$$$$$!
        !@@!@@!##########!
        !@@!@@!$$$$$$$$$$!
        !!!!!!!!!!!!!!!!!!
        !@@!@@!$$$$$$$$$$!
        !@@!@@!##########!
        !@@!@@!$$$$$$$$$$!
        !!!!!!!!!!!!!!!!!!
    """
    Grid(GridConfig(obs_radius=2, size=4, num_agents=24, map=grid_map))
    with pytest.raises(OverflowError):
        Grid(GridConfig(obs_radius=2, size=4, num_agents=25, map=grid_map))

    grid_map = """
        !!!!!!!!!!!
        !@@!@@!$$$$
        !@@!@@!####
        !@@!@@!$$$$
        !!!!!!!!!!!
        !@@!@@!$$$$
        !@@!@@!####
        !@@!@@!$$$$
        !!!!!!!!!!!
    """
    Grid(GridConfig(obs_radius=2, num_agents=16, map=grid_map))
    with pytest.raises(OverflowError):
        Grid(GridConfig(obs_radius=2, num_agents=17, map=grid_map))

    grid_map = """
            !!!!!!!!!!!
            !@@!@@!.Ab.
            !@@!@@!####
            !@@!@@!.aB.

        """
    with pytest.raises(KeyError):
        Grid(GridConfig(obs_radius=2, map=grid_map))


def test_restricted_grid():
    grid = """
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
           !@@!@@!$$$$$$$$$$!$$$$$$$$$$!$$$$$$$$$$!@@!@@!
           !@@!@@!##########!##########!##########!@@!@@!
           !@@!@@!$$$$$$$$$$!$$$$$$$$$$!$$$$$$$$$$!@@!@@!
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
           """
    env = pogema_v0(grid_config=GridConfig(map=grid, num_agents=24, seed=0, obs_radius=2))
    env.reset()

    with pytest.raises(OverflowError):
        env = pogema_v0(grid_config=GridConfig(map=grid, num_agents=25, seed=0, obs_radius=2))
        env.reset()
