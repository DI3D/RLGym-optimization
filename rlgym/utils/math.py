"""
A basic library for useful mathematical operations.
"""

from typing import Union, List
import math
import numpy as np
import numba
import random
import rlgym.rlgym_rust as rlgym_rust


def get_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.subtract(x, y)


def get_dist_1d(x: List, y: List) -> List:
    return [i-j for i, j in zip(x, y)]


def vector_projection_1d(vec: List, dest_vec: List, mag_squared=None) -> List:
    """optimized efficiency for 1d lists"""
    if mag_squared is None:
        norm = vecmag_1d(dest_vec)
        if norm == 0:
            return dest_vec
        mag_squared = norm * norm

    if mag_squared == 0:
        return dest_vec

    # dot = np.dot(vec, dest_vec)
    dot = sum([(i*j) for i, j in zip(vec, dest_vec)])
    # projection = np.multiply(np.divide(dot, mag_squared), dest_vec)
    projection = [i*j for i, j in zip([x/y for x, y in zip(dot, mag_squared)], dest_vec)]
    return projection


def vector_projection(vec: np.ndarray, dest_vec: np.ndarray, mag_squared=None) -> np.ndarray:
    if mag_squared is None:
        norm = vecmag(dest_vec)
        if norm == 0:
            return dest_vec
        mag_squared = norm * norm

    if mag_squared == 0:
        return dest_vec

    dot = np.dot(vec, dest_vec)
    projection = np.multiply(np.divide(dot, mag_squared), dest_vec)
    return projection


def scalar_projection_1d(vec: List, dest_vec: List) -> Union[List, float]:
    """optimized efficiency for 1d lists"""
    norm = vecmag_1d(dest_vec)

    if norm == 0:
        return 0

    dot = sum([(i*j) for i, j in zip(vec, dest_vec)])/norm
    return dot


def scalar_projection(vec: np.ndarray, dest_vec: np.ndarray) -> Union[np.ndarray, float]:
    norm = vecmag(dest_vec)

    if norm == 0:
        return 0

    dot = np.dot(vec, dest_vec) / norm
    return dot


def squared_vecmag_1d(vec: List) -> float:
    x = math.sqrt(sum([x * x for x in vec]))
    # x = np.linalg.norm(vec)
    return x * x


def squared_vecmag(vec: np.ndarray) -> float:
    x = np.linalg.norm(vec)
    return x * x


def norm_1d(vec: List) -> float:
    """optimized efficiency for 1d lists"""
    # assuming non-complex 1d list
    # norm = math.sqrt(sum([x*x for x in vec]))
    # norm = rlgym_rust.norm_func(vec)
    # norm = np.linalg.norm(vec)
    return rlgym_rust.norm_func(vec)


def vecmag_1d(vec: List) -> float:
    """optimized efficiency for 1d lists"""
    # norm = np.linalg.norm(vec)
    norm = norm_1d(vec)
    return norm


def vecmag(vec: np.ndarray) -> float:
    norm = np.linalg.norm(vec)
    return norm


def unitvec_1d(vec: List) -> List:
    # return np.divide(vec, vecmag(vec))
    vecm = vecmag_1d(vec)
    return [i/vecm for i in vec]


def unitvec(vec) -> np.ndarray:
    return np.divide(vec, vecmag(vec))


def cosine_similarity_1d(a, b) -> float:
    # return np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))
    a_norm = math.sqrt(sum([x * x for x in a]))
    b_norm = math.sqrt(sum([x * x for x in b]))
    return sum([i*j for i, j in zip([x/a_norm for x in a], [x/b_norm for x in b])])


def cosine_similarity(a, b) -> Union[np.ndarray, float]:
    return np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))


# @numba.njit(cache=True)
def quat_to_euler(quat) -> List:
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    sinp = 2 * (w * y - z * x)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # roll = np.arctan2(sinr_cosp, cosr_cosp)
    if abs(sinp) > 1:
        pitch = math.pi / 2
    else:
        # pitch = np.arcsin(sinp)
        pitch = math.asin(sinp)
    # yaw = np.arctan2(siny_cosp, cosy_cosp)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return [-pitch, yaw, -roll]


# From RLUtilities
# @numba.njit(cache=True)
def quat_to_rot_mtx(quat) -> np.ndarray:
    w = -quat[0]
    x = -quat[1]
    y = -quat[2]
    z = -quat[3]

    theta = np.zeros((3, 3))

    norm = np.dot(quat, quat)
    if norm != 0:
        s = 1.0 / norm

        # front direction
        theta[0, 0] = 1.0 - 2.0 * s * (y * y + z * z)
        theta[1, 0] = 2.0 * s * (x * y + z * w)
        theta[2, 0] = 2.0 * s * (x * z - y * w)

        # left direction
        theta[0, 1] = 2.0 * s * (x * y - z * w)
        theta[1, 1] = 1.0 - 2.0 * s * (x * x + z * z)
        theta[2, 1] = 2.0 * s * (y * z + x * w)

        # up direction
        theta[0, 2] = 2.0 * s * (x * z + y * w)
        theta[1, 2] = 2.0 * s * (y * z - x * w)
        theta[2, 2] = 1.0 - 2.0 * s * (x * x + y * y)

    return theta


def quat_to_rot_mtx_1d(quat) -> np.ndarray:
    # theta = rlgym_rust.quat_to_rot_mtx(quat)
    # w = -quat[0]
    # x = -quat[1]
    # y = -quat[2]
    # z = -quat[3]
    #
    # theta = np.zeros((3, 3))
    #
    # # norm = np.dot(quat, quat)
    # norm = sum([i*j for i, j in zip(quat, quat)])
    # if norm != 0:
    #     s = 1.0 / norm
    #     # s = [1.0/i for i in norm]
    #
    #     # front direction
    #     theta[0, 0] = 1.0 - 2.0 * s * (y * y + z * z)
    #     theta[1, 0] = 2.0 * s * (x * y + z * w)
    #     theta[2, 0] = 2.0 * s * (x * z - y * w)
    #
    #     # left direction
    #     theta[0, 1] = 2.0 * s * (x * y - z * w)
    #     theta[1, 1] = 1.0 - 2.0 * s * (x * x + z * z)
    #     theta[2, 1] = 2.0 * s * (y * z + x * w)
    #
    #     # up direction
    #     theta[0, 2] = 2.0 * s * (x * z + y * w)
    #     theta[1, 2] = 2.0 * s * (y * z - x * w)
    #     theta[2, 2] = 1.0 - 2.0 * s * (x * x + y * y)
    #
    return rlgym_rust.quat_to_rot_mtx(quat)


@numba.njit(cache=True)
def rotation_to_quaternion(m: np.ndarray) -> np.ndarray:
    trace = np.trace(m)
    q = np.zeros(4)

    if trace > 0:
        s = (trace + 1) ** 0.5
        q[0] = s * 0.5
        s = 0.5 / s
        q[1] = (m[2, 1] - m[1, 2]) * s
        q[2] = (m[0, 2] - m[2, 0]) * s
        q[3] = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
            s = (1 + m[0, 0] - m[1, 1] - m[2, 2]) ** 0.5
            inv_s = 0.5 / s
            q[1] = 0.5 * s
            q[2] = (m[1, 0] + m[0, 1]) * inv_s
            q[3] = (m[2, 0] + m[0, 2]) * inv_s
            q[0] = (m[2, 1] - m[1, 2]) * inv_s
        elif m[1, 1] > m[2, 2]:
            s = (1 + m[1, 1] - m[0, 0] - m[2, 2]) ** 0.5
            inv_s = 0.5 / s
            q[1] = (m[0, 1] + m[1, 0]) * inv_s
            q[2] = 0.5 * s
            q[3] = (m[1, 2] + m[2, 1]) * inv_s
            q[0] = (m[0, 2] - m[2, 0]) * inv_s
        else:
            s = (1 + m[2, 2] - m[0, 0] - m[1, 1]) ** 0.5
            inv_s = 0.5 / s
            q[1] = (m[0, 2] + m[2, 0]) * inv_s
            q[2] = (m[1, 2] + m[2, 1]) * inv_s
            q[3] = 0.5 * s
            q[0] = (m[1, 0] - m[0, 1]) * inv_s

    # q[[0, 1, 2, 3]] = q[[3, 0, 1, 2]]

    return -q


@numba.njit(cache=True)
def euler_to_rotation(pyr) -> np.ndarray:
    cp, cy, cr = np.cos(pyr)
    sp, sy, sr = np.sin(pyr)

    theta = np.zeros((3, 3))

    # front
    theta[0, 0] = cp * cy
    theta[1, 0] = cp * sy
    theta[2, 0] = sp

    # left
    theta[0, 1] = cy * sp * sr - cr * sy
    theta[1, 1] = sy * sp * sr + cr * cy
    theta[2, 1] = -cp * sr

    # up
    theta[0, 2] = -cr * cy * sp - sr * sy
    theta[1, 2] = -cr * sy * sp + sr * cy
    theta[2, 2] = cp * cr

    return theta


def rand_uvec3(rng: np.random.Generator = np.random) -> np.ndarray:
    vec: np.ndarray = rng.random(3) - 0.5
    return vec / np.linalg.norm(vec)


def rand_uvec3_1d() -> List:
    # vec = rng.random(3) - 0.5
    vec = [random.random() - 0.5 for i in range(3)]
    # return vec / np.linalg.norm(vec)
    norm_vec = norm_1d(vec)
    return [i/norm_vec for i in vec]


def rand_vec3_1d(max_norm: float) -> List:
    partial = random.random() * max_norm
    return [i*partial for i in rand_uvec3_1d()]


def rand_vec3(max_norm, rng: np.random.Generator = np.random) -> np.ndarray:
    return rand_uvec3(rng) * (rng.random() * max_norm)
