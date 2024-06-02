class SvgObject:
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


class RectangleHref(SvgObject):
    """
    Rectangle class for the SVG.
    """
    tag = 'use'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attributes['y'] = -self.attributes['y'] - self.attributes['height']
        self.attributes['href'] = "#obstacle"
        del self.attributes['height']


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
