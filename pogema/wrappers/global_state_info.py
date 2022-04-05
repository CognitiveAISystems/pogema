import gym


class GlobalStateInfo(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def get_obstacles(self, ignore_borders=False):
        return self.env.grid.get_obstacles(ignore_borders=ignore_borders)

    def get_agents_xy(self, only_active=False, ignore_borders=False):
        return self.env.grid.get_agents_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_targets_xy(self, only_active=False, ignore_borders=False):
        return self.env.grid.get_targets_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_state(self, ignore_borders=False, as_dict=False):
        return self.env.grid.get_state(ignore_borders=ignore_borders, as_dict=as_dict)
