class AgentsTargetsSizeError(Exception):
    pass


def check_grid(obstacles, agents_xy, targets_xy):
    if bool(agents_xy) != bool(targets_xy):
        raise AgentsTargetsSizeError("Agents and targets must be defined together/undefined together!")

    if not agents_xy or not targets_xy:
        return

    if len(agents_xy) != len(targets_xy):
        raise IndexError("Can't create task. Please provide agents_xy and targets_xy of the same size.")

    # check overlapping of agents
    for i in range(len(agents_xy)):
        for j in range(i + 1, len(agents_xy)):
            if agents_xy[i] == agents_xy[j]:
                raise ValueError(f"Agents can't overlap! {agents_xy[i]} is in both {i} and {j} position.")

    for start_xy, finish_xy in zip(agents_xy, targets_xy):
        s_x, s_y = start_xy
        if obstacles[s_x, s_y]:
            raise KeyError(f'Cell is {s_x, s_y} occupied by obstacle.')
        f_x, f_y = finish_xy
        if obstacles[f_x, f_y]:
            raise KeyError(f'Cell is {f_x, f_y} occupied by obstacle.')

    # todo check connectivity of starts and finishes
