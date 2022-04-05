<div align="center">


[![Pogema logo](https://raw.githubusercontent.com/Tviskaron/pogema-pics/main/pogema-logo.svg)](https://github.com/AIRI-Institute/pogema)    

**Partially-Observable Grid Environment for Multiple Agents**

[![CodeFactor](https://www.codefactor.io/repository/github/tviskaron/pogema/badge)](https://www.codefactor.io/repository/github/tviskaron/pogema)
[![Downloads](https://pepy.tech/badge/pogema)](https://pepy.tech/project/pogema)
[![CI](https://github.com/AIRI-Institute/pogema/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/AIRI-Institute/pogema/actions/workflows/CI.yml) 
[![CodeQL](https://github.com/AIRI-Institute/pogema/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/AIRI-Institute/pogema/actions/workflows/codeql-analysis.yml)    
    
</div> 

Partially observable multi-agent pathfinding (PO-MAPF) is a challenging problem which fundamentally differs from regular MAPF, in which a central controller is assumed to construct a joint plan for all agents before they start execution. PO-MAPF is intrisically decentralized and decision making (e.g. planning) here is interleaved with the execution. At each time step an agent receives a (local) observation of the environment and decides which action to take. The ultimate goal for the agents is to reach their goals while avoiding collisions with each other and the static obstacles.

POGEMA stands for Partially-Observable Grid Environment for Multiple Agents. This is a grid-based environment that was specifically designed to be flexible, tunable and scalable. It can be tailored to a variety of PO-MAPF settings. Currently the (somewhat) standard setting is supported: agents can move between the cardinally-adjacent cells of the grid, each action (move or wait) takes one time step. No information sharing between the agents is happening.

POGEMA can generate random maps and start/goals locations for the agents. It also can take custom maps as the input.

## Installation

Just install from PyPI:

```pip install pogema```

## Using Example

```python
import gym
import pogema

env = gym.make("Pogema-8x8-hard-v0")

obs = env.reset()

done = [False, ...]

while not all(done):
    # Use random policy to make actions
    obs, reward, done, info = env.step([env.action_space.sample() for _ in range(env.get_grid_config().num_agents)])
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19dSEGTQeM3oVJtVjpC162t1XApmv6APc?usp=sharing) 

## Environments

| Environment | agents density  | num agents  |  horizon    |
| -------------------------- | ----- | ----- | ---- |
| Pogema-8x8-easy-v0         | 2.2%  |   1   |  64  |
| Pogema-8x8-normal-v0       | 4.5%  |   2   |  64  |
| Pogema-8x8-hard-v0         | 8.9%  |   4   |  64  |
| Pogema-8x8-extra-hard-v0   | 17.8% |   8   |  64  |
| Pogema-16x16-easy-v0       | 2.2%  |   4   |  128 |
| Pogema-16x16-normal-v0     | 4.5%  |   8   |  128 |
| Pogema-16x16-hard-v0       | 8.9%  |   16  |  128 |
| Pogema-16x16-extra-hard-v0 | 17.8% |   32  |  128 |
| Pogema-32x32-easy-v0       | 2.2%  |   16  |  256 |
| Pogema-32x32-normal-v0     | 4.5%  |   32  |  256 |
| Pogema-32x32-hard-v0       | 8.9%  |   64  |  256 |
| Pogema-32x32-extra-hard-v0 | 17.8% |   128 |  256 |
| Pogema-64x64-easy-v0       | 2.2%  |   64  |  512 |
| Pogema-64x64-normal-v0     | 4.5%  |   128 |  512 |
| Pogema-64x64-hard-v0       | 8.9%  |   256 |  512 |
| Pogema-64x64-extra-hard-v0 | 17.8% |   512 |  512 |   

## Interfaces
Pogema provides integrations with a range of MARL frameworks: PettingZoo, PyMARL and SampleFactory. 

### PettingZoo

```python
import gym
import pogema

# Create Pogema environment with PettingZoo interface
env = gym.make("Pogema-8x8-hard-v0", integration="PettingZoo")
```

### PyMARL

```python
import gym
import pogema

env = gym.make("Pogema-8x8-hard-v0", integration="PyMARL")
```

### SampleFactory

```python
import gym
import pogema

env = gym.make("Pogema-8x8-hard-v0", integration="SampleFactory")
```

### Classic Gym


```python
import gym
import pogema

# This interface is suitable only for 
# single-agent partially observable pathfinding tasks
env = gym.make("Pogema-8x8-easy-v0", integration="SampleFactory")
```


## Customization

### Random maps
```python
import gym
from pogema import GridConfig

# Define random configuration
grid_config = GridConfig(num_agents=4,  # number of agents
                         size=8, # size of the grid
                         density=0.4,  # obstacle density
                         seed=1,  # set to None for random 
                                  # obstacles, agents and targets 
                                  # positions at each reset
                         max_episode_steps=128,  # horizon
                         obs_radius=3,  # defines field of view
                         )

env = gym.make('Pogema-v0', grid_config=grid_config)
env.reset()
env.render()

```

### Custom maps
```python
import gym
from pogema import GridConfig

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

# Define new configuration with 8 randomly placed agents
grid_config = GridConfig(map=grid, num_agents=8)

# Create custom Pogema environment
env = gym.make('Pogema-v0', grid_config=grid_config)
```




## Citation
If you use this repository in your research or wish to cite it, please make a reference to our IEEE paper: 
```
@article{skrynnik2021hybrid,
  title={Hybrid Policy Learning for Multi-Agent Pathfinding},
  author={Skrynnik, Alexey and Yakovleva, Alexandra and Davydov, Vasilii and Yakovlev, Konstantin and Panov, Aleksandr I},
  journal={IEEE Access},
  volume={9},
  pages={126034--126047},
  year={2021},
  publisher={IEEE}
}
```
We are also planning to write a separate paper (pre-print) dedicated to POGEMA entirely. The reference will appear here soon.
