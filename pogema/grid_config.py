import sys
from typing import Optional, Union
from pydantic import BaseModel, validator

from pogema.utils import CommonSettings

from typing_extensions import Literal


class GridConfig(CommonSettings, ):
    on_target: Literal['finish', 'nothing', 'restart'] = 'finish'
    seed: Optional[int] = None
    size: int = 8
    density: float = 0.3
    num_agents: int = 1
    obs_radius: int = 5
    agents_xy: Optional[list] = None
    targets_xy: Optional[list] = None
    possible_agents_xy: Optional[list] = None
    possible_targets_xy: Optional[list] = None
    collision_system: Literal['block_both', 'priority', 'soft'] = 'priority'
    persistent: bool = False
    observation_type: Literal['POMAPF', 'MAPF', 'default'] = 'default'
    map: Union[list, str] = None

    map_name: str = None

    integration: Literal['SampleFactory', 'PyMARL', 'rllib', 'gym', 'PettingZoo'] = None
    max_episode_steps: int = 64
    auto_reset: Optional[bool] = None

    @validator('seed')
    def seed_initialization(cls, v):
        assert v is None or (0 <= v < sys.maxsize), "seed must be in [0, " + str(sys.maxsize) + ']'
        return v

    @validator('size')
    def size_restrictions(cls, v):
        assert 2 <= v <= 1024, "size must be in [2, 1024]"
        return v

    @validator('density')
    def density_restrictions(cls, v):
        assert 0.0 <= v <= 1, "density must be in [0, 1]"
        return v

    @validator('num_agents')
    def num_agents_must_be_positive(cls, v):
        assert 1 <= v <= 10000, "num_agents must be in [1, 10000]"
        return v

    @validator('obs_radius')
    def obs_radius_must_be_positive(cls, v):
        assert 1 <= v <= 128, "obs_radius must be in [1, 128]"
        return v

    @validator('map', always=True)
    def map_validation(cls, v, values):
        if v is None:
            return None
        if isinstance(v, str):
            v, agents_xy, targets_xy, possible_agents_xy, possible_targets_xy = cls.str_map_to_list(v, values['FREE'],
                                                                                                    values['OBSTACLE'])
            if agents_xy and targets_xy and values.get('agents_xy') is not None and values.get(
                    'targets_xy') is not None:
                raise KeyError("""Can't create task. Please provide agents_xy and targets_xy only once.
                Either with parameters or with a map.""")
            if (agents_xy or targets_xy) and (possible_agents_xy or possible_targets_xy):
                raise KeyError("""Can't create task. Mark either possible locations or precise ones.""")
            elif agents_xy and targets_xy:
                values['agents_xy'] = agents_xy
                values['targets_xy'] = targets_xy
                values['num_agents'] = len(agents_xy)
            elif (values.get('agents_xy') is None or values.get(
                    'targets_xy') is None) and possible_agents_xy and possible_targets_xy:
                values['possible_agents_xy'] = possible_agents_xy
                values['possible_targets_xy'] = possible_targets_xy
        size = len(v)
        area = 0
        for line in v:
            size = max(size, len(line))
            area += len(line)
        values['size'] = size
        values['density'] = sum([sum(line) for line in v]) / area
        return v

    @validator('agents_xy')
    def agents_xy_validation(cls, v, values):
        if v is not None:
            cls.check_positions(v, values['size'])
            values['num_agents'] = len(v)
        return v

    @validator('targets_xy')
    def targets_xy_validation(cls, v, values):
        if v is not None:
            cls.check_positions(v, values['size'])
            values['num_agents'] = len(v)
        return v

    @validator('possible_agents_xy')
    def possible_agents_xy_validation(cls, v):
        return v

    @validator('possible_targets_xy')
    def possible_targets_xy_validation(cls, v):
        return v

    @staticmethod
    def check_positions(v, size):
        for position in v:
            x, y = position
            if not (0 <= x < size and 0 <= y < size):
                raise IndexError("Position is out of bounds!")

    @staticmethod
    def str_map_to_list(str_map, free, obstacle):
        obstacles = []
        agents = {}
        targets = {}
        possible_agents_xy = []
        possible_targets_xy = []
        special_chars = {'@', '$', '!'}

        for row_idx, line in enumerate(str_map.split()):
            row = []
            for col_idx, char in enumerate(line):
                position = (row_idx, col_idx)

                if char == '.':
                    row.append(free)
                    possible_agents_xy.append(position)
                    possible_targets_xy.append(position)
                elif char == '#':
                    row.append(obstacle)
                elif char in special_chars:
                    row.append(free)
                    if char == '@':
                        possible_agents_xy.append(position)
                    elif char == '$':
                        possible_targets_xy.append(position)
                elif 'A' <= char <= 'Z':
                    targets[char.lower()] = position
                    row.append(free)
                    possible_agents_xy.append(position)
                    possible_targets_xy.append(position)
                elif 'a' <= char <= 'z':
                    agents[char.lower()] = position
                    row.append(free)
                    possible_agents_xy.append(position)
                    possible_targets_xy.append(position)
                else:
                    raise KeyError(f"Unsupported symbol '{char}' at line {row_idx}")

            if row:
                assert len(obstacles[-1]) == len(row) if obstacles else True, f"Wrong string size for row {row_idx};"
                obstacles.append(row)

        agents_xy = [[x, y] for _, (x, y) in sorted(agents.items())]
        targets_xy = [[x, y] for _, (x, y) in sorted(targets.items())]

        assert len(targets_xy) == len(agents_xy), "Mismatch in number of agents and targets."

        if not any(char in special_chars for char in str_map):
            possible_agents_xy, possible_targets_xy = None, None

        return obstacles, agents_xy, targets_xy, possible_agents_xy, possible_targets_xy


class PredefinedDifficultyConfig(GridConfig):
    density: float = 0.3
    collision_system: Literal['priority'] = 'priority'
    obs_radius: Literal[5] = 5
    observation_type: Literal['default'] = 'default'

    @validator('density', always=True)
    def density_restrictions(cls, v):
        assert 0.299999 <= v <= 0.3000001, "density for that predefined configuration must be equal to 0.3"
        return v


class Easy8x8(PredefinedDifficultyConfig):
    size: Literal[8] = 8
    max_episode_steps: Literal[64] = 64
    num_agents: Literal[1] = 1
    map_name: Literal['Easy8x8'] = 'Easy8x8'


class Normal8x8(PredefinedDifficultyConfig):
    size: Literal[8] = 8
    max_episode_steps: Literal[64] = 64
    num_agents: Literal[2] = 2
    map_name: Literal['Normal8x8'] = 'Normal8x8'


class Hard8x8(PredefinedDifficultyConfig):
    size: Literal[8] = 8
    max_episode_steps: Literal[64] = 64
    num_agents: Literal[4] = 4
    map_name: Literal['Hard8x8'] = 'Hard8x8'


class ExtraHard8x8(PredefinedDifficultyConfig):
    size: Literal[8] = 8
    max_episode_steps: Literal[64] = 64
    num_agents: Literal[8] = 8
    map_name: Literal['ExtraHard8x8'] = 'ExtraHard8x8'


class Easy16x16(PredefinedDifficultyConfig):
    size: Literal[16] = 16
    max_episode_steps: Literal[128] = 128
    num_agents: Literal[4] = 4
    map_name: Literal['Easy16x16'] = 'Easy16x16'


class Normal16x16(PredefinedDifficultyConfig):
    size: Literal[16] = 16
    max_episode_steps: Literal[128] = 128
    num_agents: Literal[8] = 8
    map_name: Literal['Normal16x16'] = 'Normal16x16'


class Hard16x16(PredefinedDifficultyConfig):
    size: Literal[16] = 16
    max_episode_steps: Literal[128] = 128
    num_agents: Literal[16] = 16
    map_name: Literal['Hard16x16'] = 'Hard16x16'


class ExtraHard16x16(PredefinedDifficultyConfig):
    size: Literal[16] = 16
    max_episode_steps: Literal[128] = 128
    num_agents: Literal[32] = 32
    map_name: Literal['ExtraHard16x16'] = 'ExtraHard16x16'


class Easy32x32(PredefinedDifficultyConfig):
    size: Literal[32] = 32
    max_episode_steps: Literal[256] = 256
    num_agents: Literal[16] = 16
    map_name: Literal['Easy32x32'] = 'Easy32x32'


class Normal32x32(PredefinedDifficultyConfig):
    size: Literal[32] = 32
    max_episode_steps: Literal[256] = 256
    num_agents: Literal[32] = 32
    map_name: Literal['Normal32x32'] = 'Normal32x32'


class Hard32x32(PredefinedDifficultyConfig):
    size: Literal[32] = 32
    max_episode_steps: Literal[256] = 256
    num_agents: Literal[64] = 64
    map_name: Literal['Hard32x32'] = 'Hard32x32'


class ExtraHard32x32(PredefinedDifficultyConfig):
    size: Literal[32] = 32
    max_episode_steps: Literal[256] = 256
    num_agents: Literal[128] = 128
    map_name: Literal['ExtraHard32x32'] = 'ExtraHard32x32'


class Easy64x64(PredefinedDifficultyConfig):
    size: Literal[32] = 64
    max_episode_steps: Literal[512] = 512
    num_agents: Literal[16] = 64
    map_name: Literal['Easy64x64'] = 'Easy64x64'


class Normal64x64(PredefinedDifficultyConfig):
    size: Literal[32] = 64
    max_episode_steps: Literal[512] = 512
    num_agents: Literal[16] = 128
    map_name: Literal['Normal64x64'] = 'Normal64x64'


class Hard64x64(PredefinedDifficultyConfig):
    size: Literal[32] = 64
    max_episode_steps: Literal[512] = 512
    num_agents: Literal[16] = 256
    map_name: Literal['Hard64x64'] = 'Hard64x64'


class ExtraHard64x64(PredefinedDifficultyConfig):
    size: Literal[32] = 64
    max_episode_steps: Literal[512] = 512
    num_agents: Literal[16] = 512
    map_name: Literal['ExtraHard64x64'] = 'ExtraHard64x64'
