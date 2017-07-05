import numpy as np
from scene_node import SceneNode
from common_utils import *

class Sphere(SceneNode):
    def __init__(self, position, r =1.0, color=[1, 0, 0], transparency=0.1, reflection=0.2, refraction=1.0, n=1.0):
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
        OS = origin - self.position

        a = np.dot(direction, direction)
        b = 2 * np.dot(direction, OS)
        c = np.dot(OS, OS) - self.r * self.r
        disc = b * b - 4 * a * c

        if disc > 0:
            distSqrt = np.sqrt(disc)
            t0 = (-b - distSqrt) / (2.0 * a)
            t1 = (-b + distSqrt) / (2.0 * a)
            t0, t1 = min(t0, t1), max(t0, t1)
            if t1 >= 10 * epsilon and t0 < epsilon:
                return True, t1, normalize(origin + direction * t1 - self.position)
            elif t0 >= 10 * epsilon:
                return True, t0, normalize(origin + direction * t0 - self.position)

        return False, np.inf, np.zeros(3)