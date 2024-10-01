<div align="center">

**Partially-Observable Grid Environment for Multiple Agents**
    
</div> 

Partially Observable Multi-Agent Pathfinding (PO-MAPF) is a challenging problem that fundamentally differs from regular MAPF. In regular MAPF, a central controller constructs a joint plan for all agents before they start execution. However, PO-MAPF is intrinsically decentralized, and decision-making, such as planning, is interleaved with execution. At each time step, an agent receives a local observation of the environment and decides which action to take. The ultimate goal for the agents is to reach their goals while avoiding collisions with each other and the static obstacles.

POGEMA stands for Partially-Observable Grid Environment for Multiple Agents. It is a grid-based environment that was specifically designed to be flexible, tunable, and scalable. It can be tailored to a variety of PO-MAPF settings. Currently, the somewhat standard setting is supported, in which agents can move between the cardinal-adjacent cells of the grid, and each action (move or wait) takes one time step. No information sharing occurs between the agents. POGEMA can generate random maps and start/goal locations for the agents. It also accepts custom maps as input.

## Installation

```python setup install```

## Using Example

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(grid_config=GridConfig())

obs, info = env.reset()

while True:
    # Using random policy to make actions
    obs, reward, terminated, truncated, info = env.step(env.sample_actions())
    env.render()
    if all(terminated) or all(truncated):
        break
```

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
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(integration="gymnasium"))
```



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
