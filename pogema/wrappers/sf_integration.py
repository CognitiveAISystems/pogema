import gym


class IsMultiAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.is_multiagent = True

    @property
    def num_agents(self):
        return self.env.get_num_agents()


class AutoResetWrapper(gym.Wrapper):
    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)
        if all(dones):
            observations = self.env.reset()
        return observations, rewards, dones, infos
