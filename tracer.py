#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import argparse

from plane import Plane
from sphere import Sphere

from common_utils import *

# Defines
depth_max = 3
light_depth_max = 3
shadow_steps = 8
h = 768
w = 1024
r = float(w) / h

# Light
L = np.array([0, 0.9 ,0.0])
color_light = np.zeros(3)
ambient = 0.3
diffuse_c = 0.3
specular_c = 0.2
specular_k = 30

# Camera
O = np.array([0., 0.0, -3.])# Camera. O = np.array([0., 0.35, -1.])# Camera.
Q = np.array([-1., 0.0, -1.0])  # Camera pointing to.

# Screen coordinates: x0, y0, x1, y1.
screen = (-1.0, -1.0 / r , 1.0, 1.0 / r )

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
    dL = np.linalg.norm(L - M)

    color = np.array(obj.color)
    toL = normalize(L - M)
    toO = normalize(O - M)

    # shadow
    num = shadow_steps
    mult = num

    for k in range(num):
        for i, s_obj in enumerate(scene):
            if i == obj_idx:
                continue

            rayOl = M + N * 0.0001 + np.random.uniform(0, 1e-6, 3)
            rayDl = toL

            is_intersect, t_obj, norm = s_obj.hit(rayOl, rayDl)

            if is_intersect and t_obj < np.inf:
                ML = rayOl + rayDl * t_obj
                if np.linalg.norm(M - ML) <= dL:
                    mult -= 1
                    # return None, np.zeros(3), np.zeros(3), np.zeros(3)
                    continue
    # mult = num

    # Color
    col_ray = ambient
    
    # Lambert shading
    col_ray += obj.diffuse_c * max(np.dot(N, toL), 0) * color

    # Blinn-Phong shading
    col_ray += obj.specular_c * max(np.dot(N, normalize(toL + toO)), 0) ** specular_k * color_light
    
    return obj, M, N, col_ray * (mult / float(num))


def create_scene():
    # scene.append(Plane([1.2, 0.0, 0.0], [-1.0, 0.0, 0.0], color=[0.8, 0.3, 0.4]))
    scene.append(Plane([-1.5, 0.0, 0.0], [1.0, 0.0, 0.0], color=[1.0, 0.0, 0.0], reflection=0.1)) #left
    scene.append(Plane([ 1.5, 0.0, 0.0], [-1.0, 0.0, 0.0], color=[1.0, 0.0, 0.0], reflection=0.1)) #right
    scene.append(Plane([ 0.0, -1.0, 0.0], [0.0, 1.0, 0.0], color=[0.0, 0.0, 1.0], reflection=0.05)) #floor
    scene.append(Plane([ 0.0, 1.0, 0.0], [0.0, -1.0, 0.0], color=[0.0, 0.0, 1.0], reflection=0.05)) #ceil
    scene.append(Plane([ 0.0, 0.0, 2.5], [0.0, 0.0, -1.0], color=[0.0, 1.0, 0.0], reflection=0.1)) #far
    # scene.append(Plane([-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], color=[0.4, 0.3, 0.4]))

    scene.append(Sphere([0.0, -0.6, 0.3], r=0.2, color=[1.0, 0.1, 0.1], transparency=0.3, reflection=0.3))
    scene.append(Sphere([-0.7, -0.5, 0.5], r=0.3, color=[0.1, 1.0, 0.1], transparency=0.2, reflection=0.4))
    scene.append(Sphere([-0.4, 0.3, 1.2], r=0.2, color=[0.1, 1.0, 0.1], transparency=0.2, reflection=0.4))
    scene.append(Sphere([0.5, -0.5, 1.5], r=0.3, color=[0.1, 1.0, 0.1], transparency=0.2, reflection=0.4))
    
    # scene.append(Sphere([0.6, 0.4, -0.1], r=0.2, color=[0.0, 1.0, 0.1], transparency=0.6, n=1.5))
    # scene.append(Sphere([-0.6, 0.6, 0.5], r=0.4, color=[0.0, 1.0, 1.1], transparency=0.4, reflection=0.4, n=1.3))
 

def refract(v, n, q):
    q = 2.0 - q;
    cosi = np.dot(n, v);
    o = (v * q - n * (-cosi + q * cosi));
    return o


def ray_trace(params, rayO, rayD, reflection, refraction, depth, n1 = 1.0):
    obj, M, N, col_ray = trace_ray(rayO, rayD)
    if not obj:
        return np.zeros(3)

    n = obj.n
    transparency = obj.transparency
    
    if depth > params.max_depth:
        return transparency * col_ray

    rayOrefl = M + N * 0.0001
    rayOrefr = M - N * 0.00001

    rayDrefl = normalize(rayD - 2 * np.dot(rayD, N) * N)
    rayDrefr = refract(rayD, N, n1 / n)

    refr = refraction * obj.refraction
    refl = reflection * obj.reflection

    if refl > epsilon:
        refl_color = refl * ray_trace(params, rayOrefl, rayDrefl, refl, refr, depth + 1, n)
    else:
        refl_color = 0.0

    if refr > epsilon:
        refr_color = refr * ray_trace(params, rayOrefr, rayDrefr, refl, refr, depth + 1, n)
    else:
        refr_color = 0.0

    return refr_color + refl_color + obj.transparency * col_ray


def ray_trace_worker(inp, params):
    """
    A worker instance
    :param inp: input parameters
    :return: Color for the current pixel
    """
    i, x, j, y = inp

    Q[:2] = (x, y)
    D = normalize(Q - params.O)

    return np.clip(ray_trace(params, O, D, 1, 1, 1), 0, 1)


def main():
    """
    Application entry point
    """
    parser = argparse.ArgumentParser(description='Python ray tracer')

    parser.add_argument(
        '--workers', type=int, default=4,
        help='Number of ray tracing workers'
    )

    parser.add_argument(
        '--max_depth', type=int, default=3,
        help='Recursion depth'
    )

    parser.add_argument(
        '--height', type=int, default=64,
        help='An image height'
    )

    parser.add_argument(
        '--width', type=int, default=128,
        help='An image width'
    )

    parser.add_argument(
        '--image', type=str, default='output.png',
        help='A destination image'
    )

    parser.add_argument(
        '--show-incomplete', dest='show_incomplete', action='store_true', default=False,
        help='Render intermediate results to the image'
    )

    parser.add_argument(
        '--O', type=float, nargs='+', default=[-0.2, 0.0, -3.4],
        help='Camera position'
    )

    # Parse command line arguments
    params = parser.parse_args()

    # Create the scene
    create_scene()

    # Create multiprocessing pool
    pool = mp.Pool(processes=params.workers)

    # Create empty buffer image
    img = np.zeros((params.height, params.width, 3))

    # Parallel by the image columns
    for i, x in enumerate(np.linspace(screen[0], screen[2], params.width)):
        if i % 5 == 0:
            print(i / float(params.width) * 100, "%")
            if params.show_incomplete:
                plt.imsave(params.image, img)

        # Create pool parameters
        inputs = [(i, x, j, y) for j, y in enumerate(np.linspace(screen[1], screen[3], params.height))]

        # Parallel evaluation
        row = pool.map(partial(ray_trace_worker, params=params), inputs)
        img[:, i] = np.flip(row, axis=0)

    # Save results
    plt.imsave(params.image, img)


if __name__ == '__main__':
    main()