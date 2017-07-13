from scene_node import SceneNode
from common_utils import *


class Sphere(SceneNode):
    def __init__(self, position, r=1.0, color=[1, 0, 0], transparency=0.1, reflection=0.2, refraction=0.2, n=1.0):
        """
        Sphere constructor.
        :param position: Sphere position
        :param r: Sphere radius
        :param color: Color
        :param transparency: Transparency (0.0 is a fully transparent)
        :param reflection: Reflection rate
        :param refraction: Refraction rate
        :param n: Refraction coefficient
        """
        self.position = np.array(position)
        self.color = np.array(color)
        self.transparency = transparency
        self.refraction = refraction
        self.reflection = reflection
        self.n = n
        self.r = r
        self.diffuse_c = 0.5
        self.specular_c = 0.8

    def hit(self, origin, direction):
        """

        :param origin: Ray origin
        :param direction: Ray direction
        :return: [has intersection, t, normal]
        """
        os = origin - self.position

        a = np.dot(direction, direction)
        b = 2 * np.dot(direction, os)
        c = np.dot(os, os) - self.r * self.r
        disc = b * b - 4 * a * c

        if disc >= 0:
            dist_sqrt = np.sqrt(disc)
            t0 = (-b - dist_sqrt) / (2.0 * a)
            t1 = (-b + dist_sqrt) / (2.0 * a)
            t0, t1 = min(t0, t1), max(t0, t1)

            if t1 >= 10 * epsilon and t0 < epsilon:
                return True, t1, normalize(origin + direction * t1 - self.position)
            elif t0 >= 10 * epsilon:
                return True, t0, normalize(origin + direction * t0 - self.position)

        return False, np.inf, np.zeros(3)
