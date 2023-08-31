<div align="center">


[![Pogema logo](https://raw.githubusercontent.com/Tviskaron/pogema-pics/main/pogema-logo.svg)](https://github.com/AIRI-Institute/pogema)    

**Partially-Observable Grid Environment for Multiple Agents**

[![CodeFactor](https://www.codefactor.io/repository/github/tviskaron/pogema/badge)](https://www.codefactor.io/repository/github/tviskaron/pogema)
[![Downloads](https://static.pepy.tech/badge/pogema)](https://pepy.tech/project/pogema)
[![CI](https://github.com/AIRI-Institute/pogema/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/AIRI-Institute/pogema/actions/workflows/CI.yml) 
[![CodeQL](https://github.com/AIRI-Institute/pogema/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/AIRI-Institute/pogema/actions/workflows/codeql-analysis.yml)    
    
</div> 

Partially Observable Multi-Agent Pathfinding (PO-MAPF) is a challenging problem that fundamentally differs from regular MAPF. In regular MAPF, a central controller constructs a joint plan for all agents before they start execution. However, PO-MAPF is intrinsically decentralized, and decision-making, such as planning, is interleaved with execution. At each time step, an agent receives a local observation of the environment and decides which action to take. The ultimate goal for the agents is to reach their goals while avoiding collisions with each other and the static obstacles.

POGEMA stands for Partially-Observable Grid Environment for Multiple Agents. It is a grid-based environment that was specifically designed to be flexible, tunable, and scalable. It can be tailored to a variety of PO-MAPF settings. Currently, the somewhat standard setting is supported, in which agents can move between the cardinal-adjacent cells of the grid, and each action (move or wait) takes one time step. No information sharing occurs between the agents. POGEMA can generate random maps and start/goal locations for the agents. It also accepts custom maps as input.

## Installation

Just install from PyPI:

```pip install pogema```

## Using Example

```python
from pogema import pogema_v0, Hard8x8

env = pogema_v0(grid_config=Hard8x8())

obs, info = env.reset()

while True:
    # Using random policy to make actions
    obs, reward, terminated, truncated, info = env.step(env.sample_actions())
    env.render()
    if all(terminated) or all(truncated):
        break

```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19dSEGTQeM3oVJtVjpC162t1XApmv6APc?usp=sharing) 

## Environments

| Config | agents density  | num agents  |  horizon    |
| ----------------- | ----- | ----- | ---- |
| Easy8x8           | 2.2%  |   1   |  64  |
| Normal8x8         | 4.5%  |   2   |  64  |
| Hard8x8           | 8.9%  |   4   |  64  |
| ExtraHard8x8      | 17.8% |   8   |  64  |
| Easy16x16         | 2.2%  |   4   |  128 |
| Normal16x16       | 4.5%  |   8   |  128 |
| Hard16x16         | 8.9%  |   16  |  128 |
| ExtraHard16x16    | 17.8% |   32  |  128 |
| Easy32x32         | 2.2%  |   16  |  256 |
| Normal32x32       | 4.5%  |   32  |  256 |
| Hard32x32         | 8.9%  |   64  |  256 |
| ExtraHard32x32    | 17.8% |   128 |  256 |
| Easy64x64         | 2.2%  |   64  |  512 |
| Normal64x64       | 4.5%  |   128 |  512 |
| Hard64x64         | 8.9%  |   256 |  512 |
| ExtraHard64x64    | 17.8% |   512 |  512 |   

## Baselines 
The [baseline implementations](https://github.com/Tviskaron/pogema-baselines) are available as a separate repository.

## Interfaces
Pogema provides integrations with a range of MARL frameworks: PettingZoo, PyMARL and SampleFactory. 

### PettingZoo

```python
from pogema import pogema_v0, GridConfig

# Create Pogema environment with PettingZoo interface
env = pogema_v0(GridConfig(integration="PettingZoo"))
```

### PyMARL

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(integration="PyMARL"))
```

### SampleFactory

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(integration="SampleFactory"))
```

### Gymnasium

Pogema is fully capable for single-agent pathfinding tasks. 

```python
import gymnasium as gym
import pogema

# This interface provides experience only for agent with id=0,
# other agents will take random actions.
env = gym.make("Pogema-v0")
```

Example of training [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) DQN to solve single-agent pathfinding tasks: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vPwTd0PnzpWrB-bCHqoLSVwU9G9Lgcmv?usp=sharing)




## Customization

### Random maps
```python
from pogema import pogema_v0, GridConfig

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

env = pogema_v0(grid_config=grid_config)
env.reset()
env.render()

```

### Custom maps
```python
from pogema import pogema_v0, GridConfig

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
env = pogema_v0(grid_config=grid_config)
```




## Citation
If you use this repository in your research or wish to cite it, please make a reference to our paper: 
```
@misc{https://doi.org/10.48550/arxiv.2206.10944,
  doi = {10.48550/ARXIV.2206.10944},  
  url = {https://arxiv.org/abs/2206.10944},
  author = {Skrynnik, Alexey and Andreychuk, Anton and Yakovlev, Konstantin and Panov, Aleksandr I.},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Multiagent Systems (cs.MA), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {POGEMA: Partially Observable Grid Environment for Multiple Agents},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
