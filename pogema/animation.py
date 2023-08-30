import os
import typing
from itertools import cycle
from gymnasium import logger, Wrapper

from pydantic import BaseModel

from pogema import GridConfig, pogema_v0
from pogema.grid import Grid
from pogema.wrappers.persistence import PersistentWrapper, AgentState


class AnimationSettings(BaseModel):
    """
    Settings for the animation.
    """
    r: int = 35
    stroke_width: int = 10
    scale_size: int = 100
    time_scale: float = 0.28
    draw_start: int = 100
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


class AnimationConfig(BaseModel):
    """
    Configuration for the animation.
    """
    directory: str = 'renders/'
    static: bool = False
    show_agents: bool = True
    egocentric_idx: typing.Optional[int] = None
    uid: typing.Optional[str] = None
    save_every_idx_episode: typing.Optional[int] = 1
    show_border: bool = True
    show_lines: bool = False


class GridHolder(BaseModel):
    """
    Holds the grid and the history.
    """
    obstacles: typing.Any = None
    episode_length: int = None
    height: int = None
    width: int = None
    colors: dict = None
    history: list = None


class SvgObject:
    """
    Main class for the SVG.
    """
    tag = None

    def __init__(self, **kwargs):
        self.attributes = kwargs
        self.animations = []

    def add_animation(self, animation):
        self.animations.append(animation)

    @staticmethod
    def render_attributes(attributes):
        result = " ".join([f'{x.replace("_", "-")}="{y}"' for x, y in sorted(attributes.items())])
        return result

    def render(self):
        animations = '\n'.join([a.render() for a in self.animations]) if self.animations else None
        if animations:
            return f"<{self.tag} {self.render_attributes(self.attributes)}> {animations} </{self.tag}>"
        return f"<{self.tag} {self.render_attributes(self.attributes)} />"


class Rectangle(SvgObject):
    """
    Rectangle class for the SVG.
    """
    tag = 'rect'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attributes['y'] = -self.attributes['y'] - self.attributes['height']


class Circle(SvgObject):
    """
    Circle class for the SVG.
    """
    tag = 'circle'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attributes['cy'] = -self.attributes['cy']


class Line(SvgObject):
    """
    Line class for the SVG.
    """
    tag = 'line'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attributes['y1'] = -self.attributes['y1']
        self.attributes['y2'] = -self.attributes['y2']


class Animation(SvgObject):
    """
    Animation class for the SVG.
    """
    tag = 'animate'

    def render(self):
        return f"<{self.tag} {self.render_attributes(self.attributes)}/>"


class Drawing:
    """
    Drawing, analog of the DrawSvg class in the pogema package.
    """

    def __init__(self, height, width, display_inline=False, origin=(0, 0)):
        self.height = height
        self.width = width
        self.display_inline = display_inline
        self.origin = origin
        self.elements = []

    def add_element(self, element):
        self.elements.append(element)

    def render(self):
        view_box = (0, -self.height, self.width, self.height)
        results = [f'''<?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
             width="{self.width // 10}" height="{self.height // 10}" viewBox="{" ".join(map(str, view_box))}">''',
                   '\n<defs>\n', '</defs>\n']
        for element in self.elements:
            results.append(element.render())
        results.append('</svg>')
        return "\n".join(results)


class AnimationMonitor(Wrapper):
    """
    Defines the animation, which saves the episode as SVG.
    """

    def __init__(self, env, animation_config=AnimationConfig()):
        # Wrapping env using PersistenceWrapper for saving the history.
        env = PersistentWrapper(env)
        super().__init__(env)

        self.history = self.env.get_history()

        self.svg_settings: AnimationSettings = AnimationSettings()
        self.animation_config: AnimationConfig = animation_config

        self._episode_idx = 0

    def step(self, action):
        """
        Saves information about the episode.
        :param action: current actions
        :return: obs, reward, done, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        multi_agent_terminated = isinstance(terminated, (list, tuple)) and all(terminated)
        single_agent_terminated = isinstance(terminated, (bool, int)) and terminated
        multi_agent_truncated = isinstance(truncated, (list, tuple)) and all(truncated)
        single_agent_truncated = isinstance(truncated, (bool, int)) and truncated

        if multi_agent_terminated or single_agent_terminated or multi_agent_truncated or single_agent_truncated:
            save_tau = self.animation_config.save_every_idx_episode
            if save_tau:
                if (self._episode_idx + 1) % save_tau or save_tau == 1:
                    if not os.path.exists(self.animation_config.directory):
                        logger.info(f"Creating pogema monitor directory {self.animation_config.directory}", )
                        os.makedirs(self.animation_config.directory, exist_ok=True)

                    path = os.path.join(self.animation_config.directory,
                                        self.pick_name(self.grid_config, self._episode_idx))
                    self.save_animation(path)

        return obs, reward, terminated, truncated, info

    @staticmethod
    def pick_name(grid_config: GridConfig, episode_idx=None, zfill_ep=5):
        """
        Picks a name for the SVG file.
        :param grid_config: configuration of the grid
        :param episode_idx: idx of the episode
        :param zfill_ep: zfill for the episode number
        :return:
        """
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
        """
        Resets the environment and resets the current positions of agents and targets
        :param kwargs:
        :return: obs: observation
        """
        obs = self.env.reset(**kwargs)

        self._episode_idx += 1
        self.history = self.env.get_history()

        return obs

    def create_animation(self, animation_config=None):
        """
        Creates the animation.
        :param animation_config: configuration of the animation
        :return: drawing: drawing object
        """
        anim_cfg = animation_config
        if anim_cfg is None:
            anim_cfg = self.animation_config

        grid: Grid = self.grid
        cfg = self.svg_settings
        colors = cycle(cfg.colors)
        agents_colors = {index: next(colors) for index in range(self.grid_config.num_agents)}

        if anim_cfg.egocentric_idx is not None:
            anim_cfg.egocentric_idx %= self.grid_config.num_agents

        decompressed_history: list[list[AgentState]] = self.env.decompress_history(self.history)

        # Change episode length for egocentric environment
        if anim_cfg.egocentric_idx is not None:
            episode_length = decompressed_history[anim_cfg.egocentric_idx][-1].step + 1
            for agent_idx in range(self.grid_config.num_agents):
                decompressed_history[agent_idx] = decompressed_history[agent_idx][:episode_length]
        else:
            episode_length = len(decompressed_history[0])

        # Add last observation one more time to highlight the final state
        for agent_idx in range(self.grid_config.num_agents):
            decompressed_history[agent_idx].append(decompressed_history[agent_idx][-1])

        # Change episode length for static environment
        if anim_cfg.static:
            episode_length = 1
            decompressed_history = [[decompressed_history[idx][-1]] for idx in range(len(decompressed_history))]

        gh = GridHolder(width=len(grid.obstacles), height=len(grid.obstacles[0]),
                        obstacles=grid.obstacles,
                        colors=agents_colors,
                        episode_length=episode_length,
                        history=decompressed_history, )

        render_width, render_height = gh.height * cfg.scale_size + cfg.scale_size, gh.width * cfg.scale_size + cfg.scale_size

        drawing = Drawing(width=render_width, height=render_height, display_inline=False, origin=(0, 0))
        obstacles = self.create_obstacles(gh, anim_cfg)

        agents = []
        targets = []

        if anim_cfg.show_agents:
            agents = self.create_agents(gh, anim_cfg)
            targets = self.create_targets(gh, anim_cfg)

            if not anim_cfg.static:
                self.animate_agents(agents, anim_cfg.egocentric_idx, gh)
                self.animate_targets(targets, gh, anim_cfg)
        if anim_cfg.show_lines:
            grid_lines = self.create_grid_lines(gh, anim_cfg, render_width, render_height)
            for line in grid_lines:
                drawing.add_element(line)
        for obj in [*obstacles, *agents, *targets, ]:
            drawing.add_element(obj)

        if anim_cfg.egocentric_idx is not None:
            field_of_view = self.create_field_of_view(grid_holder=gh, animation_config=anim_cfg)
            if not anim_cfg.static:
                self.animate_obstacles(obstacles=obstacles, grid_holder=gh, animation_config=anim_cfg)
                self.animate_field_of_view(field_of_view, anim_cfg.egocentric_idx, gh)
            drawing.add_element(field_of_view)

        return drawing

    def create_grid_lines(self, grid_holder: GridHolder, animation_config: AnimationConfig, render_width,
                          render_height):
        """
        Creates the grid lines.
        :param grid_holder: grid holder
        :param animation_config: animation configuration
        :return: grid_lines: list of grid lines
        """
        cfg = self.svg_settings
        grid_lines = []
        for i in range(-1, grid_holder.height + 1):
            # vertical lines
            x0 = x1 = i * cfg.scale_size + cfg.scale_size / 2
            y0 = 0
            y1 = render_height
            grid_lines.append(
                Line(x1=x0, y1=y0, x2=x1, y2=y1, stroke=cfg.obstacle_color, stroke_width=cfg.stroke_width // 1.5))
        for i in range(-1, grid_holder.width + 1):
            # continue
            # horizontal lines
            x0 = 0
            y0 = y1 = i * cfg.scale_size + cfg.scale_size / 2
            x1 = render_width
            grid_lines.append(
                Line(x1=x0, y1=y0, x2=x1, y2=y1, stroke=cfg.obstacle_color, stroke_width=cfg.stroke_width // 1.5))

        # for i in range(grid_holder.width):
        #     grid_lines.append(Line(start=(0, i * cfg.scale_size),
        #                            end=(grid_holder.height * cfg.scale_size, i * cfg.scale_size),
        #                            stroke=cfg.grid_color, stroke_width=cfg.grid_width))
        return grid_lines

    def save_animation(self, name='render.svg', animation_config: typing.Optional[AnimationConfig] = None):
        """
        Saves the animation.
        :param name: name of the file
        :param animation_config: animation configuration
        :return: None
        """
        animation = self.create_animation(animation_config)
        with open(name, "w") as f:
            f.write(animation.render())

    @staticmethod
    def fix_point(x, y, length):
        """
        Fixes the point to the grid.
        :param x: coordinate x
        :param y: coordinate y
        :param length: size of the grid
        :return: x, y: fixed coordinates
        """
        return length - y - 1, x

    @staticmethod
    def check_in_radius(x1, y1, x2, y2, r) -> bool:
        """
        Checks if the point is in the radius.
        :param x1: coordinate x1
        :param y1: coordinate y1
        :param x2: coordinate x2
        :param y2: coordinate y2
        :param r: radius
        :return:
        """
        return x2 - r <= x1 <= x2 + r and y2 - r <= y1 <= y2 + r

    def create_field_of_view(self, grid_holder, animation_config):
        """
        Creates the field of view for the egocentric agent.
        :param grid_holder:
        :param animation_config:
        :return:
        """
        cfg = self.svg_settings
        gh: GridHolder = grid_holder
        ego_idx = animation_config.egocentric_idx
        x, y = gh.history[ego_idx][0].get_xy()
        cx = cfg.draw_start + y * cfg.scale_size
        cy = cfg.draw_start + (gh.width - x - 1) * cfg.scale_size

        dr = (self.grid_config.obs_radius + 1) * cfg.scale_size - cfg.stroke_width * 2
        result = Rectangle(x=cx - dr + cfg.r, y=cy - dr + cfg.r,
                           width=2 * dr - 2 * cfg.r, height=2 * dr - 2 * cfg.r,
                           stroke=cfg.ego_color, stroke_width=cfg.stroke_width,
                           fill='none',
                           rx=cfg.rx, stroke_dasharray=cfg.stroke_dasharray,
                           )

        return result

    def animate_field_of_view(self, view, agent_idx, grid_holder):
        """
        Animates the field of view.
        :param view:
        :param agent_idx:
        :param grid_holder:
        :return:
        """
        gh: GridHolder = grid_holder
        cfg = self.svg_settings
        x_path = []
        y_path = []
        for state in gh.history[agent_idx]:
            x, y = state.get_xy()
            dr = (self.grid_config.obs_radius + 1) * cfg.scale_size - cfg.stroke_width * 2
            cx = cfg.draw_start + y * cfg.scale_size
            cy = -cfg.draw_start + -(gh.width - x - 1) * cfg.scale_size
            x_path.append(str(cx - dr + cfg.r))
            y_path.append(str(cy - dr + cfg.r))

        visibility = ['visible' if state.is_active() else 'hidden' for state in gh.history[agent_idx]]

        view.add_animation(self.compressed_anim('x', x_path, cfg.time_scale))
        view.add_animation(self.compressed_anim('y', y_path, cfg.time_scale))
        view.add_animation(self.compressed_anim('visibility', visibility, cfg.time_scale))

    def animate_agents(self, agents, egocentric_idx, grid_holder):
        """
        Animates the agents.
        :param agents:
        :param egocentric_idx:
        :param grid_holder:
        :return:
        """
        gh: GridHolder = grid_holder
        cfg = self.svg_settings
        for agent_idx, agent in enumerate(agents):
            x_path = []
            y_path = []
            opacity = []
            for agent_state in gh.history[agent_idx]:
                x, y = agent_state.get_xy()

                x_path.append(str(cfg.draw_start + y * cfg.scale_size))
                y_path.append(str(-cfg.draw_start + -(gh.width - x - 1) * cfg.scale_size))

                if egocentric_idx is not None:
                    ego_x, ego_y = agent_state.get_xy()
                    if self.check_in_radius(x, y, ego_x, ego_y, self.grid_config.obs_radius):
                        opacity.append('1.0')
                    else:
                        opacity.append(str(cfg.shaded_opacity))

            visibility = ['visible' if state.is_active() else 'hidden' for state in gh.history[agent_idx]]

            agent.add_animation(self.compressed_anim('cy', y_path, cfg.time_scale))
            agent.add_animation(self.compressed_anim('cx', x_path, cfg.time_scale))
            agent.add_animation(self.compressed_anim('visibility', visibility, cfg.time_scale))
            if opacity:
                agent.add_animation(self.compressed_anim('opacity', opacity, cfg.time_scale))

    @classmethod
    def compressed_anim(cls, attr_name, tokens, time_scale, rep_cnt='indefinite'):
        """
        Compresses the animation.
        :param attr_name:
        :param tokens:
        :param time_scale:
        :param rep_cnt:
        :return:
        """
        tokens, times = cls.compress_tokens(tokens)
        cumulative = [0, ]
        for t in times:
            cumulative.append(cumulative[-1] + t)
        times = [str(round(value / cumulative[-1], 10)) for value in cumulative]
        tokens = [tokens[0]] + tokens

        times = times
        tokens = tokens
        return Animation(attributeName=attr_name,
                         dur=f'{time_scale * (-1 + cumulative[-1])}s',
                         values=";".join(tokens),
                         repeatCount=rep_cnt,
                         keyTimes=";".join(times))

    @staticmethod
    def wisely_add(token, cnt, tokens, times):
        """
        Adds the token to the tokens and times.
        :param token:
        :param cnt:
        :param tokens:
        :param times:
        :return:
        """
        if cnt > 1:
            tokens += [token, token]
            times += [1, cnt - 1]
        else:
            tokens.append(token)
            times.append(cnt)

    @classmethod
    def compress_tokens(cls, input_tokens: list):
        """
        Compresses the tokens.
        :param input_tokens:
        :return:
        """
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

    def animate_targets(self, targets, grid_holder, animation_config):
        """
        Animates the targets.
        :param targets:
        :param grid_holder:
        :param animation_config:
        :return:
        """
        gh: GridHolder = grid_holder
        cfg = self.svg_settings
        ego_idx = animation_config.egocentric_idx

        for agent_idx, target in enumerate(targets):
            target_idx = ego_idx if ego_idx is not None else agent_idx

            x_path = []
            y_path = []

            for step_idx, state in enumerate(gh.history[target_idx]):
                x, y = state.get_target_xy()
                x_path.append(str(cfg.draw_start + y * cfg.scale_size))
                y_path.append(str(-cfg.draw_start + -(gh.width - x - 1) * cfg.scale_size))

            visibility = ['visible' if state.is_active() else 'hidden' for state in gh.history[agent_idx]]

            if self.grid_config.on_target == 'restart':
                target.add_animation(self.compressed_anim('cy', y_path, cfg.time_scale))
                target.add_animation(self.compressed_anim('cx', x_path, cfg.time_scale))
            target.add_animation(self.compressed_anim("visibility", visibility, cfg.time_scale))

    def create_obstacles(self, grid_holder, animation_config):
        """
        Creates the obstacles.
        :param grid_holder:
        :param animation_config:
        :return:
        """
        gh = grid_holder
        cfg = self.svg_settings

        result = []
        r = self.grid_config.obs_radius
        for i in range(gh.height):
            for j in range(gh.width):
                x, y = self.fix_point(i, j, gh.width)
                if not animation_config.show_border:
                    if i == r - 1 or j == r - 1 or j == gh.width - r or i == gh.height - r:
                        continue
                if gh.obstacles[x][y] != self.grid_config.FREE:
                    obs_settings = {}
                    obs_settings.update(x=cfg.draw_start + i * cfg.scale_size - cfg.r,
                                        y=cfg.draw_start + j * cfg.scale_size - cfg.r,
                                        width=cfg.r * 2,
                                        height=cfg.r * 2,
                                        rx=cfg.rx,
                                        fill=self.svg_settings.obstacle_color)

                    if animation_config.egocentric_idx is not None and cfg.egocentric_shaded:
                        initial_positions = [agent_states[0].get_xy() for agent_states in gh.history]
                        ego_x, ego_y = initial_positions[animation_config.egocentric_idx]
                        if not self.check_in_radius(x, y, ego_x, ego_y, self.grid_config.obs_radius):
                            obs_settings.update(opacity=cfg.shaded_opacity)

                    result.append(Rectangle(**obs_settings))

        return result

    def animate_obstacles(self, obstacles, grid_holder, animation_config):
        """

        :param obstacles:
        :param grid_holder:
        :param animation_config:
        :return:
        """
        gh: GridHolder = grid_holder
        obstacle_idx = 0
        cfg = self.svg_settings

        for i in range(gh.height):
            for j in range(gh.width):
                x, y = self.fix_point(i, j, gh.width)
                if gh.obstacles[x][y] == self.grid_config.FREE:
                    continue
                opacity = []
                seen = set()
                for step_idx, agent_state in enumerate(gh.history[animation_config.egocentric_idx]):
                    ego_x, ego_y = agent_state.get_xy()
                    if self.check_in_radius(x, y, ego_x, ego_y, self.grid_config.obs_radius):
                        seen.add((x, y))
                    if (x, y) in seen:
                        opacity.append(str(1.0))
                    else:
                        opacity.append(str(cfg.shaded_opacity))

                obstacle = obstacles[obstacle_idx]
                obstacle.add_animation(self.compressed_anim('opacity', opacity, cfg.time_scale))

                obstacle_idx += 1

    def create_agents(self, grid_holder, animation_config):
        """
        Creates the agents.
        :param grid_holder:
        :param animation_config:
        :return:
        """
        gh: GridHolder = grid_holder
        cfg = self.svg_settings

        agents = []
        initial_positions = [agent_states[0].get_xy() for agent_states in gh.history]
        for idx, (x, y) in enumerate(initial_positions):

            if not any([agent_state.is_active() for agent_state in gh.history[idx]]):
                continue

            circle_settings = {}
            circle_settings.update(cx=cfg.draw_start + y * cfg.scale_size,
                                   cy=cfg.draw_start + (gh.width - x - 1) * cfg.scale_size,
                                   r=cfg.r, fill=gh.colors[idx])
            ego_idx = animation_config.egocentric_idx
            if ego_idx is not None:
                ego_x, ego_y = initial_positions[ego_idx]
                if not self.check_in_radius(x, y, ego_x, ego_y, self.grid_config.obs_radius) and cfg.egocentric_shaded:
                    circle_settings.update(opacity=cfg.shaded_opacity)
                if ego_idx == idx:
                    circle_settings.update(fill=self.svg_settings.ego_color)
                else:
                    circle_settings.update(fill=self.svg_settings.ego_other_color)
            agent = Circle(**circle_settings)
            agents.append(agent)

        return agents

    def create_targets(self, grid_holder, animation_config):
        """
        Creates the targets.
        :param grid_holder:
        :param animation_config:
        :return:
        """
        gh: GridHolder = grid_holder
        cfg = self.svg_settings
        targets = []
        for agent_idx, agent_states in enumerate(gh.history):

            tx, ty = agent_states[0].get_target_xy()
            x, y = ty, gh.width - tx - 1

            if not any([agent_state.is_active() for agent_state in gh.history[agent_idx]]):
                continue

            circle_settings = {}
            circle_settings.update(cx=cfg.draw_start + x * cfg.scale_size,
                                   cy=cfg.draw_start + y * cfg.scale_size,
                                   r=cfg.r,
                                   stroke=gh.colors[agent_idx], stroke_width=cfg.stroke_width, fill='none')
            if animation_config.egocentric_idx is not None:
                if animation_config.egocentric_idx != agent_idx:
                    continue

                circle_settings.update(stroke=cfg.ego_color)
            target = Circle(**circle_settings)
            targets.append(target)
        return targets


def main():
    grid_config = GridConfig(size=8, num_agents=5, obs_radius=2, seed=9, on_target='finish', max_episode_steps=128)
    env = pogema_v0(grid_config=grid_config)
    env = AnimationMonitor(env)

    env.reset()
    done = [False]

    while not all(done):
        _, _, done, _ = env.step(env.sample_actions())

    env.save_animation('out-static.svg', AnimationConfig(static=True, save_every_idx_episode=None))
    env.save_animation('out-static-ego.svg', AnimationConfig(egocentric_idx=0, static=True))
    env.save_animation('out-static-no-agents.svg', AnimationConfig(show_agents=False, static=True))
    env.save_animation("out.svg")
    env.save_animation("out-ego.svg", AnimationConfig(egocentric_idx=0))


if __name__ == '__main__':
    main()
