import numpy as np
from scene_node import SceneNode
from common_utils import *

class Plane(SceneNode):
    def __init__(self, position, normal, color=[0.4, 0.4, 0.4], transparency=1.0, reflection=0.2, refraction=0, n=1.0):
        """

        :param position: Plane position
        :param normal: Plane normal (ex. [0, 1, 0])
        :param color: Plane color
        :param transparency: Plane transparency
        :param reflection: Plane reflection rate
        :param refraction: Plane refraction rate
        :param n: Plane refraction coefficient
        """
        self.position = np.array(position)
        self.color = np.array(color)
        self.normal = np.array(normal)
        self.refraction = refraction
        self.transparency = transparency
        self.reflection = reflection
        self.n = n
        self.diffuse_c = 0.95
        self.specular_c = 0.58

    def hit(self, origin, direction):
        """

        :param origin: Ray origin
        :param direction: Ray direction
        :return: [has intersection, t, normal]
        """
        denom = np.dot(direction, self.normal)
        if np.abs(denom) < epsilon:
            return False, np.inf, np.zeros(3)

        d = np.dot(self.position - origin, self.normal) / denom

        if d < 0:
            return False, np.inf, np.zeros(3)

        return True, d, self.normal
