import math

import numpy as np


def calculate_fov_matrix_size(ray_length: int, half_fov_angle: float) -> tuple:
    """
    Calculate the size of the field of view (FOV) matrix.

    This function computes the width of the FOV matrix based on the given ray length and half of the FOV angle.
    The width is adjusted to be an even number if necessary.

    Args:
        ray_length (int): The length of the ray.
        half_fov_angle (float): Half of the field of view angle in radians.

    Returns:
        tuple: A tuple containing the ray length and the calculated matrix width.
    """

    matrix_width = int(2 * math.sin(half_fov_angle) * ray_length)

    matrix_width = matrix_width if matrix_width % 2 == 0 else matrix_width + 1

    return (ray_length, matrix_width)


def step_angle(fov_angle: float, number_of_rays: int):
    return fov_angle / number_of_rays
