import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp

from plane import Plane
from sphere import Sphere
from cylinder import Cylinder

from common_utils import *

# Defines
depth_max = 5
light_depth_max = 1
h = 768
w = 1024
r = float(w) / h

# Light
L = np.array([3., 3., -5.])
color_light = np.zeros(3)
ambient = 0.0
diffuse_c = 0.5
specular_c = 0.5
specular_k = 70

# Camera
O = np.array([0., 0.35, -1.])# Camera.
Q = np.zeros(3)  # Camera pointing to.

# Screen coordinates: x0, y0, x1, y1.
screen = (-1.0, -1.0 / r + 0.25, 1.0, 1.0 / r + 0.25)

# Scene 
scene = []
    
def trace_ray(rayO, rayD, depth=0):
    if depth > light_depth_max:
        return None, np.zeros(3), np.zeros(3), np.zeros(3)

    t = np.inf
    obj_idx = -1
    N = []

    for i, obj in enumerate(scene):
        is_intersect, t_obj, norm = obj.hit(rayO, rayD)
        if is_intersect and t_obj <= t:
            t, obj, N = t_obj, obj, norm
            obj_idx = i

    if t == np.inf or obj_idx < 0:
        return None, np.zeros(3), np.zeros(3), np.zeros(3)

    obj = scene[obj_idx]
    
    M = rayO + rayD * t

    color = np.array(obj.color)
    toL = normalize(L - M)
    toO = normalize(O - M)

    # shadow
    num = 3
    mult = num
    for k in range(num):
        for i, s_obj in enumerate(scene):
            if i == obj_idx:
                continue

            is_intersect, t_obj, norm = s_obj.hit(M + N * 0.001, L + np.random.uniform(-1e-3, 1e-3, 3))
            if is_intersect and t_obj < np.inf:
                mult -= 1
                continue

    # Color
    col_ray = ambient
    
    # Lambert shading
    col_ray += obj.diffuse_c * max(np.dot(N, toL), 0) * color

    # Blinn-Phong shading
    col_ray += obj.specular_c * max(np.dot(N, normalize(toL + toO)), 0) ** specular_k * color_light
    
    return obj, M, N, col_ray * (mult / float(num))


def create_scene():
    scene.append(Plane([0, 0, 0], [0.0, 1.0, 0.0], color=[0.4, 0.3, 0.4]))
    scene.append(Sphere([0.4, 0.5, 0.8], r=0.5, color=[1., 0., 0.], transparency=1.0))
    scene.append(Sphere([-0.1, 0.4, -0.2], r=0.3, color=[0.1, 0.1, 0.1], transparency=0.1, n=2.5))
    scene.append(Sphere([0.6, 0.4, -0.1], r=0.2, color=[0.0, 1.0, 0.1], transparency=0.6, n=1.5))
    scene.append(Sphere([-0.6, 0.6, 0.5], r=0.4, color=[0.0, 1.0, 1.1], transparency=0.4, reflection=0.4, n=1.3))
 

def refract(v, n, q):
    q = 2.0 - q;
    cosi = np.dot(n, v);
    o = (v * q - n * (-cosi + q * cosi));
    return o


def ray_trace(rayO, rayD, reflection, refraction, depth, n1 = 1.0):
    obj, M, N, col_ray = trace_ray(rayO, rayD)
    if not obj:
        return np.zeros(3)

    n = obj.n
    transparency = obj.transparency
    
    if depth > depth_max:
        return transparency * col_ray

    rayOrefl = M + N * 0.0001
    rayOrefr = M - N * 0.000001

    rayDrefl = normalize(rayD - 2 * np.dot(rayD, N) * N)
    rayDrefr = refract(rayD, N, n1 / n)

    refr = refraction * obj.refraction
    refl = reflection * obj.reflection
    
    refl_color = reflection * ray_trace(rayOrefl, rayDrefl, refl, refr, depth + 1, n)
    refr_color = refraction * ray_trace(rayOrefr, rayDrefr, refl, refr, depth + 1, n)

    return refr_color + refl_color + obj.transparency * col_ray


def ray_trace_worker(inp):
    i, x, j, y = inp

    Q[:2] = (x, y)
    D = normalize(Q - O)
    rayO, rayD = O, D

    return np.clip(ray_trace(O, D, 1, 1, 1), 0, 1)


if __name__ == '__main__':
    create_scene()

    pool = mp.Pool(processes=4)
    img = np.zeros((h, w, 3))
    for i, x in enumerate(np.linspace(screen[0], screen[2], w)):
        if i % 5 == 0:
            print (i / float(w) * 100, "%")
            plt.imsave('fig.png', img)
            
        inputs = np.repeat(np.array([i, x]), h)
        inputs = [(i, x, j, y) for j, y  in enumerate(np.linspace(screen[1], screen[3], h))]

        row = pool.map(ray_trace_worker, inputs)
        img[:, i] = np.flip(row, axis=0)    

    plt.imsave('fig.png', img)