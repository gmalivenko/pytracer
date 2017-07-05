import numpy as np
from scene_node import SceneNode
from common_utils import *

class Cylinder(SceneNode):
    def __init__(self, position, r =1.0, h=0.5, color=[1, 0, 0], transparency=1.0, reflection=0.2, refraction=1.0, n=1.0):
        self.position = np.array(position)
        self.color = np.array(color)
        self.transparency = transparency
        self.refraction = refraction
        self.reflection = reflection
        self.n = n
        self.h = h
        self.r = r
        self.diffuse_c = 0.5
        self.specular_c = 0.8

    def hit(self, origin, direction):
        OS = origin - self.position

        a = np.dot(direction[[0,2]], direction[[0,2]])
        b = 2 * np.dot(direction[[0,2]], OS[[0,2]])
        c = np.dot(OS[[0,2]], OS[[0,2]]) - self.r * self.r
        disc = b * b - 4 * a * c

        if disc >= 0:
            distSqrt = np.sqrt(disc)
            t0 = (-b - distSqrt) / (2.0 * a)
            t1 = (-b + distSqrt) / (2.0 * a)
            t0, t1 = min(t0, t1), max(t0, t1)

            y0 = (origin + direction * t0)[1]
            y1 = (origin + direction * t1)[1]

            min_y = self.position[1] - self.h
            max_y = self.position[1] + self.h

            t = []
            for i in [t0, t1]:
                if (origin + direction * i)[1] >= min_y and (origin + direction * i)[1] <= max_y:
                   t.append(i)

            t = np.array(t)
            res = t[t > 0]
            if (len(res) > 0):
                N = normalize(origin[[0,2]] + direction[[0,2]] * np.min(t[t > 0]) - self.position[[0,2]])
                N = np.array([N[0], 0, N[1]])
                return True, np.min(t[t > 0]), N

            

            if y1 >= max_y and y0 <= max_y:
                return True, (t1 + t0) / 2, np.array([0, 1, 0])

            if y1 >= min_y and y0 <= min_y:
                return True, (t1 + t0) / 2, np.array([0, -1, 0])

        return False, np.inf, np.zeros(3)