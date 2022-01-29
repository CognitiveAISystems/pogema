import gym


class SeedWrapper(gym.Wrapper):
    def __init__(self, env, seeds):
        self._seeds = seeds
        self._seed_cnt = 0
        super().__init__(env)

    def set_seed(self):
        self.env.config.seed = self._seeds[self._seed_cnt]
        self._seed_cnt = (self._seed_cnt + 1) % len(self._seeds)

    def reset(self, **kwargs):
        self.set_seed()
        return self.env.reset(**kwargs)
