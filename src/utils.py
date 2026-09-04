import torch
import torch.nn.functional as F


def rotation_axis(a, b):
    return torch.linalg.cross(a, b)

def rotation_angle(a, b):
    cos_theta = torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    return torch.acos(cos_theta)

def skew_symmetric(v):
    return torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=torch.float32)

def rodrigues_rotation_matrix(a, b):
    v = rotation_axis(a, b)
    theta = rotation_angle(a, b)

    if torch.norm(v) == 0:
        return torch.eye(3)

    v = v / torch.norm(v)

    v_skew = skew_symmetric(v)

    R = torch.eye(3) + torch.sin(theta) * v_skew + (1 - torch.cos(theta)) * torch.mm(v_skew, v_skew)
    return R.float()

def rotate_points(points_src: torch.tensor, a, b):
    R = rodrigues_rotation_matrix(a, b)

    rotated_points = torch.mm(points_src, R.T)
    return rotated_points, R

def rotate_points_z_axis(points_src: torch.tensor, normal):
    normal_z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    normal = normal / torch.norm(normal)

    R = rodrigues_rotation_matrix(normal, normal_z)

    rotated_points = torch.mm(points_src, R.T)
    return rotated_points, R

def fit_quadric_surface(points_src:torch.tensor):
    x = points_src[:, 0]
    y = points_src[:, 1]
    z = points_src[:, 2]

    # z = ax^2 + by^2 + cxy + dx + ey + f
    A_matrix = torch.stack([x ** 2, y ** 2, x * y, x, y, torch.ones_like(x)], dim=1)
    fit_result = torch.linalg.lstsq(A_matrix, z).solution

    fit_result = torch.reshape(fit_result, (6, 1))

    z = torch.reshape(z, (len(z), 1))
    distances = torch.mm(A_matrix, fit_result) - z

    mse = torch.mean(distances ** 2)
    total_error = torch.sum(distances ** 2)

    return fit_result, mse, total_error

def sample_from_qudric(quadric_surface:torch.tensor, points_src:torch.tensor):
    x = points_src[:, 0]
    y = points_src[:, 1]
    # z = points_src[:, 2]

    A_matrix = torch.stack([x ** 2, y ** 2, x * y, x, y, torch.ones_like(x)], dim=1)

    quadric_surface = torch.reshape(quadric_surface, (6, 1))

    new_z = torch.mm(A_matrix, quadric_surface)

    sample_points = torch.stack([x, y, new_z.flatten()], dim=1)

    return sample_points

