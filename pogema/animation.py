import os
import typing
from copy import deepcopy
from itertools import cycle
from gym import logger
import drawSvg
import gym
from pogema.wrappers.multi_time_limit import MultiTimeLimit

from pydantic import BaseModel

from pogema import GridConfig
from pogema.grid import Grid


class AnimationSettings(BaseModel):
    r: int = 35
    stroke_width: int = 10
    scale_size: int = 100
    time_scale: float = 0.28
    draw_start: int = 50
    rx: int = 15

    obstacle_color: str = '#84A1AE'
    ego_color: str = '#c1433c'
    ego_other_color: str = '#72D5C8'
    shaded_opacity: float = 0.2
    egocentric_shaded: bool = True
    stroke_dasharray: int = 25

    colors: list = [
        '#c1433c',
        '#2e6f9e',
        '#6e81af',
        '#00b9c8',
        '#72D5C8',
        '#0ea08c',
        '#8F7B66',
    ]

    directory = 'renders/'


class GridHolder(BaseModel):
    agents_xy: list = None
    agents_xy_history: list = None
    agents_done_history: list = None
    targets_xy: list = None
    obstacles: typing.Any = None
    egocentric_idx: int = None
    episode_length: int = None
    height: int = None
    width: int = None
    colors: dict = None


class AnimationMonitor(gym.Wrapper):
    def __init__(self, env, animation_settings=AnimationSettings(), egocentric_idx: int = None):
        super().__init__(env)
        self.grid_cfg = None
        self.cfg: AnimationSettings = animation_settings
        self.dones_history = None
        self.agents_xy_history = None
        self.egocentric_idx = egocentric_idx
        self._episode_idx = 0

    def step(self, action):
        obs, reward, dones, info = self.env.step(action)

        self.dones_history.append(dones)
        self.agents_xy_history.append(deepcopy(self.env.grid.positions_xy))
        if all(dones):
            if not os.path.exists(self.cfg.directory):
                logger.info(f"Creating pogema monitor directory {self.cfg.directory}", )
                os.makedirs(self.cfg.directory, exist_ok=True)
            self.save_animation(name=self.cfg.directory + self.pick_name(self.grid_cfg, self._episode_idx))

        return obs, reward, dones, info

    @staticmethod
    def pick_name(grid_config: GridConfig, episode_idx=None, zfill_ep=5):
        gc = grid_config
        name = 'pogema'
        if episode_idx is not None:
            name += f'-ep{str(episode_idx).zfill(zfill_ep)}'
        if gc:
            if gc.map_name:
                name += f'-{gc.map_name}'
            if gc.seed is not None:
                name += f'-seed{gc.seed}'
        else:
            name += '-render'
        return name + '.svg'

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self._episode_idx += 1

        self.grid_cfg: GridConfig = self.env.config
        self.dones_history = [[False for _ in range(self.env.config.num_agents)]]
        self.agents_xy_history = [deepcopy(self.env.grid.positions_xy)]
        return obs

    def create_animation(self, egocentric_idx):
        if egocentric_idx is None:
            egocentric_idx = self.egocentric_idx

        grid: Grid = self.env.grid
        cfg = self.cfg
        colors = cycle(cfg.colors)
        agents_colors = {index: next(colors) for index in range(self.grid_cfg.num_agents)}

        episode_length = len(self.dones_history)
        if egocentric_idx is not None:
            for step_idx, dones in enumerate(self.dones_history):
                if dones[egocentric_idx]:
                    episode_length = min(len(self.dones_history), step_idx + 1)
                    break
        gh = GridHolder(agents_xy=grid.positions_xy, width=len(grid.obstacles), height=len(grid.obstacles[0]),
                        agents_xy_history=self.agents_xy_history,
                        agents_done_history=self.dones_history, egocentric_idx=egocentric_idx, obstacles=grid.obstacles,
                        colors=agents_colors, targets_xy=grid.finishes_xy, episode_length=episode_length)

        render_height, render_width = gh.height * cfg.scale_size, gh.width * cfg.scale_size
        drawing = drawSvg.Drawing(render_height, render_width, displayInline=False,
                                  origin=(0, 0))
        drawing.setRenderSize(render_height // 10, render_width // 10)
        obstacles = self.create_obstacles(gh)
        agents = self.create_agents(gh)
        targets = self.create_targets(gh)

        self.animate_agents(agents, egocentric_idx, gh)
        self.animate_targets(targets, gh)

        if egocentric_idx is not None:
            self.animate_obstacles(obstacles=obstacles, egocentric_idx=egocentric_idx, grid_holder=gh)
            field_of_view = self.create_field_of_view(grid_holder=gh)
            self.animate_field_of_view(field_of_view, egocentric_idx, gh)
            drawing.append(field_of_view)

        for obj in [*obstacles, *agents, *targets]:
            drawing.append(obj)

        return drawing

    def save_animation(self, name='render.svg', egocentric_idx=None):
        animation = self.create_animation(egocentric_idx)
        animation.saveSvg(fname=name)

    @staticmethod
    def fix_point(x, y, length):
        return length - y - 1, x

    @staticmethod
    def check_in_radius(x1, y1, x2, y2, r):
        return x2 - r <= x1 <= x2 + r and y2 - r <= y1 <= y2 + r

    def create_field_of_view(self, grid_holder):
        cfg = self.cfg
        gh: GridHolder = grid_holder
        x, y = gh.agents_xy_history[0][gh.egocentric_idx] if gh.agents_xy_history else gh.agents_xy[gh.egocentric_idx]
        cx = cfg.draw_start + y * cfg.scale_size
        cy = cfg.draw_start + (gh.width - x - 1) * cfg.scale_size

        dr = (self.grid_cfg.obs_radius + 1) * cfg.scale_size - cfg.stroke_width * 2
        result = drawSvg.Rectangle(cx - dr + cfg.r,
                                   cy - dr + cfg.r,
                                   2 * dr - 2 * cfg.r,
                                   2 * dr - 2 * cfg.r,
                                   stroke=cfg.ego_color,
                                   stroke_width=cfg.stroke_width,
                                   fill='none',
                                   rx=cfg.rx,
                                   stroke_dasharray=cfg.stroke_dasharray,
                                   )

        return result

    def animate_field_of_view(self, view, agent_idx, grid_holder):
        gh: GridHolder = grid_holder
        cfg = self.cfg
        x_path = []
        y_path = []
        for agents_xy in gh.agents_xy_history[:gh.episode_length]:
            x, y = agents_xy[agent_idx]
            dr = (self.grid_cfg.obs_radius + 1) * cfg.scale_size - cfg.stroke_width * 2
            cx = cfg.draw_start + y * cfg.scale_size
            cy = -cfg.draw_start + -(gh.width - x - 1) * cfg.scale_size
            x_path.append(str(cx - dr + cfg.r))
            y_path.append(str(cy - dr + cfg.r))

        visibility = []
        for dones in gh.agents_done_history[:gh.episode_length]:
            visibility.append('hidden' if dones[agent_idx] else "visible")

        view.appendAnim(self.compressed_anim('x', x_path, cfg.time_scale))
        view.appendAnim(self.compressed_anim('y', y_path, cfg.time_scale))
        view.appendAnim(self.compressed_anim('visibility', visibility, cfg.time_scale))

    def animate_agents(self, agents, egocentric_idx, grid_holder):
        gh: GridHolder = grid_holder
        cfg = self.cfg
        for agent_idx, agent in enumerate(agents):
            x_path = []
            y_path = []
            opacity = []
            for agents_xy in gh.agents_xy_history[:gh.episode_length]:
                x, y = agents_xy[agent_idx]
                x_path.append(str(cfg.draw_start + y * cfg.scale_size))
                y_path.append(str(-cfg.draw_start + -(gh.width - x - 1) * cfg.scale_size))

                if egocentric_idx is not None:
                    ego_x, ego_y = agents_xy[egocentric_idx]
                    if self.check_in_radius(x, y, ego_x, ego_y, self.grid_cfg.obs_radius):
                        opacity.append('1.0')
                    else:
                        opacity.append(str(cfg.shaded_opacity))

            visibility = []
            for dones in gh.agents_done_history[:gh.episode_length]:
                visibility.append('hidden' if dones[agent_idx] else "visible")

            agent.appendAnim(self.compressed_anim('cy', y_path, cfg.time_scale))
            agent.appendAnim(self.compressed_anim('cx', x_path, cfg.time_scale))
            agent.appendAnim(self.compressed_anim('visibility', visibility, cfg.time_scale))
            if opacity:
                agent.appendAnim(self.compressed_anim('opacity', opacity, cfg.time_scale))

    @classmethod
    def compressed_anim(cls, attr_name, tokens, time_scale, rep_cnt='indefinite'):
        tokens, times = cls.compress_tokens(tokens)
        cumulative = [0, ]
        for t in times:
            cumulative.append(cumulative[-1] + t)
        times = [str(round(value / cumulative[-1], 10)) for value in cumulative]
        tokens = [tokens[0]] + tokens

        times = times
        tokens = tokens
        return drawSvg.Animate(attr_name, f'{time_scale * (-1 + cumulative[-1])}s', ";".join(tokens),
                               repeatCount=rep_cnt,
                               keyTimes=";".join(times))

    @staticmethod
    def wisely_add(token, cnt, tokens, times):
        if cnt > 1:
            tokens += [token, token]
            times += [1, cnt - 1]
        else:
            tokens.append(token)
            times.append(cnt)

    @classmethod
    def compress_tokens(cls, input_tokens: list):

        tokens = []
        times = []
        if input_tokens:
            cur_idx = 0
            cnt = 1
            for idx in range(1, len(input_tokens)):
                if input_tokens[idx] == input_tokens[cur_idx]:
                    cnt += 1
                else:
                    cls.wisely_add(input_tokens[cur_idx], cnt, tokens, times)
                    cnt = 1
                    cur_idx = idx
            cls.wisely_add(input_tokens[cur_idx], cnt, tokens, times)
        return tokens, times

    def animate_targets(self, targets, grid_holder):
        gh: GridHolder = grid_holder
        cfg = self.cfg
        for target_idx, target in enumerate(targets):
            if gh.egocentric_idx is not None:
                if gh.egocentric_idx != target_idx:
                    continue
            visibility = []
            for dones in gh.agents_done_history[:gh.episode_length]:
                visibility.append('hidden' if dones[target_idx] else "visible")
            target.appendAnim(self.compressed_anim("visibility", visibility, cfg.time_scale))

    def create_obstacles(self, grid_holder):
        gh = grid_holder
        cfg = self.cfg

        result = []
        for i in range(gh.height):
            for j in range(gh.width):
                x, y = self.fix_point(i, j, gh.width)
                if gh.obstacles[x][y] != self.grid_cfg.FREE:
                    obs_settings = {}
                    obs_settings.update(x=cfg.draw_start + i * cfg.scale_size - cfg.r,
                                        y=cfg.draw_start + j * cfg.scale_size - cfg.r,
                                        width=cfg.r * 2,
                                        height=cfg.r * 2,
                                        rx=cfg.rx,
                                        fill=self.cfg.obstacle_color)

                    if gh.egocentric_idx is not None and cfg.egocentric_shaded:
                        initial_positions = gh.agents_xy_history[0] if gh.agents_xy_history else gh.agents_xy
                        ego_x, ego_y = initial_positions[gh.egocentric_idx]
                        if not self.check_in_radius(x, y, ego_x, ego_y, self.grid_cfg.obs_radius):
                            obs_settings.update(opacity=cfg.shaded_opacity)

                    result.append(drawSvg.Rectangle(**obs_settings))

        return result

    def animate_obstacles(self, obstacles, egocentric_idx, grid_holder):
        gh: GridHolder = grid_holder
        obstacle_idx = 0
        cfg = self.cfg

        for i in range(gh.height):
            for j in range(gh.width):
                x, y = self.fix_point(i, j, gh.width)
                if gh.obstacles[x][y] == self.grid_cfg.FREE:
                    continue
                opacity = []
                seen = set()
                for step_idx, agents_xy in enumerate(gh.agents_xy_history[:gh.episode_length]):
                    ego_x, ego_y = agents_xy[egocentric_idx]
                    if self.check_in_radius(x, y, ego_x, ego_y, self.grid_cfg.obs_radius):
                        seen.add((x, y))
                    if (x, y) in seen:
                        opacity.append(str(1.0))
                    else:
                        opacity.append(str(cfg.shaded_opacity))

                obstacle = obstacles[obstacle_idx]
                obstacle.appendAnim(self.compressed_anim('opacity', opacity, cfg.time_scale))

                obstacle_idx += 1

    def create_agents(self, grid_holder):
        gh: GridHolder = grid_holder
        cfg = self.cfg

        agents = []
        initial_positions = gh.agents_xy_history[0] if gh.agents_xy_history else gh.agents_xy
        for idx, (x, y) in enumerate(initial_positions):
            if gh.agents_done_history[0][idx]:
                continue

            circle_settings = {}
            circle_settings.update(cx=cfg.draw_start + y * cfg.scale_size,
                                   cy=cfg.draw_start + (gh.width - x - 1) * cfg.scale_size,
                                   r=cfg.r, fill=gh.colors[idx])
            if gh.egocentric_idx is not None:
                ego_x, ego_y = initial_positions[gh.egocentric_idx]
                if not self.check_in_radius(x, y, ego_x, ego_y, self.grid_cfg.obs_radius) and cfg.egocentric_shaded:
                    circle_settings.update(opacity=cfg.shaded_opacity)
                if gh.egocentric_idx == idx:
                    circle_settings.update(fill=self.cfg.ego_color)
                else:
                    circle_settings.update(fill=self.cfg.ego_other_color)
            agent = drawSvg.Circle(**circle_settings)
            agents.append(agent)

        return agents

    def create_targets(self, grid_holder):
        gh: GridHolder = grid_holder
        cfg = self.cfg
        targets = []
        for idx, (tx, ty) in enumerate(gh.targets_xy):
            if gh.agents_done_history[0][idx]:
                continue
            x, y = ty, gh.width - tx - 1
            circle_settings = {}
            circle_settings.update(cx=cfg.draw_start + x * cfg.scale_size,
                                   cy=cfg.draw_start + y * cfg.scale_size,
                                   r=cfg.r,
                                   stroke=gh.colors[idx], stroke_width=cfg.stroke_width, fill='none')
            if gh.egocentric_idx is not None:
                if gh.egocentric_idx != idx:
                    continue

                circle_settings.update(stroke=cfg.ego_color)
            target = drawSvg.Circle(**circle_settings)
            targets.append(target)
        return targets


def main():
    grid_config = GridConfig(size=64, num_agents=256, obs_radius=2, seed=7)
    env = gym.make('Pogema-v0', config=grid_config)
    env = MultiTimeLimit(env, max_episode_steps=12)
    env = AnimationMonitor(env, egocentric_idx=None)

    env.reset()
    env.save_animation('out-static.svg', egocentric_idx=None)
    env.save_animation('out-static-ego.svg', egocentric_idx=0)
    done = [False]
    while not all(done):
        if all(done):
            break
        _, _, done, _ = env.step([env.action_space.sample() for _ in range(grid_config.num_agents)])
    env.save_animation("out.svg", egocentric_idx=None)
    env.save_animation("out-ego.svg", egocentric_idx=0)


if __name__ == '__main__':
    main()
