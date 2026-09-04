import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from scipy.ndimage import distance_transform_edt
from scipy.signal import convolve2d
import open3d as o3d
import SimpleITK as sitk


def compute_CD_L1(points1, points2):
    dist1 = np.mean(np.min(cdist(points1, points2), axis=1))
    dist2 = np.mean(np.min(cdist(points2, points1), axis=1))
    return dist1 + dist2

def compute_HD(points1, points2):
    dist = cdist(points1, points2)

    dist1 = np.min(dist, axis=0)
    dist1 = np.max(dist1)

    dist2 = np.min(dist, axis=1)
    dist2 = np.max(dist2)

    return max(dist1, dist2)

def compute_EMD(points1, points2):
    flat1 = points1.flatten()
    flat2 = points2.flatten()

    return wasserstein_distance(flat1, flat2)

def get_slice_edge(slice):
    kernel = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])

    padded = np.pad(slice, 1, mode='constant', constant_values=False)
    conv_result = convolve2d(padded.astype(int), kernel, mode='valid')

    return slice & (conv_result < 4)

def get_voxel_contours(label_voxel):
    edge_voxel = np.zeros_like(label_voxel)

    for d in range(label_voxel.shape[0]):
        edge_voxel[d] = get_slice_edge(label_voxel[d])

    return edge_voxel

def get_label_edge_points(nii_path, label_id):
    image = sitk.ReadImage(nii_path)
    volume = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()

    volume[np.where(volume != label_id)] = 0
    volume[np.where(volume == label_id)] = 1

    volume = get_voxel_contours(volume)

    label_pos = np.where(volume == 1)

    points = []
    if len(label_pos[0]) == 0:
        return np.array(points)

    points = np.zeros((len(label_pos[0]), 3), np.float32)
    points[:, 0] = -label_pos[2] * spacing[0]
    points[:, 1] = -label_pos[1] * spacing[1]
    points[:, 2] = label_pos[0] * spacing[2]

    points = points + [volume.shape[2] * spacing[0], volume.shape[1] * spacing[1], 0]

    return points

def get_mesh_sample_points(mesh_path, sample_num):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_poisson_disk(number_of_points=sample_num)
    points = np.asarray(pcd.points)

    return points


def main(label_path, mesh_path, label_id):

    points1 = get_label_edge_points(label_path, label_id)
    points2 = get_mesh_sample_points(mesh_path, sample_num=len(points1))

    chamfer = compute_CD_L1(points1, points2)
    hausdorff = compute_HD(points1, points2)
    emd = compute_EMD(points1, points2)

    return chamfer, hausdorff, emd


if __name__ == '__main__':
    label_path = ''
    mesh_path = ''
    label_id = ''

    chamfer, hausdorff, emd = main(label_path, mesh_path, label_id)

    print(chamfer, hausdorff, emd)