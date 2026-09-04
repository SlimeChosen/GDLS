import SimpleITK as sitk
import numpy as np
import torch
import distmap


class VolumeImage():
    @staticmethod
    def get_voxel_grid_from_mask(nii_path, label_id: int, voxel_size: int, resample_size, dilate=False):
        image = sitk.ReadImage(nii_path)

        if resample_size > 0:
            image = VolumeImage.resample_volumeImage(image, resample_size, True)

        volume_array = sitk.GetArrayFromImage(image)

        label_array = VolumeImage.get_label_array(volume_array, image.GetSpacing(), label_id)

        voxel_grid, translate_distance = VolumeImage.generate_voxelgrid_from_pointcloud(torch.tensor(label_array),
                                                                                        voxel_size)

        return voxel_grid, translate_distance, image.GetSpacing()

    @staticmethod
    def get_voxel_grid_edge_from_mask(nii_path, label_id:int, voxel_size:int, resample_size, dilate=False):
        image = sitk.ReadImage(nii_path)

        if resample_size > 0:
            image = VolumeImage.resample_volumeImage(image, resample_size, True)

        volume_array = sitk.GetArrayFromImage(image)

        label_array = VolumeImage.get_label_array(volume_array, image.GetSpacing(), label_id)

        voxel_grid, translate_distance = VolumeImage.generate_voxelgrid_from_pointcloud(torch.tensor(label_array), voxel_size)

        voxel_grid = VolumeImage.get_voxel_grid_edge(voxel_grid)

        return voxel_grid, translate_distance, image.GetSpacing()

    @staticmethod
    def get_voxel_grid_edge(voxel_grid:torch.tensor):
        distance = distmap.euclidean_distance_transform(voxel_grid)

        # boundary_pos = np.where(distance==1)
        boundary_pos = np.where((distance > 0) & (distance < 1.7321))

        new_voxel_grid = torch.zeros(voxel_grid.shape, dtype=torch.uint8)
        new_voxel_grid[boundary_pos[0], boundary_pos[1], boundary_pos[2]] = 1

        return new_voxel_grid

    @staticmethod
    def compute_voxel_grid_gradient(voxel_grid:torch.tensor):
        gradient = torch.gradient(voxel_grid.clone().detach().float())

        # dist = distmap.euclidean_distance_transform(voxel_grid.float(), ndim=2)
        # gradient = torch.gradient(dist, dim=[0, 1, 2])
        # gradient = [gradient[2], gradient[1], gradient[0]]

        return gradient


    @staticmethod
    def get_label_array(volume_array, spacing, label_id) -> np.array:
        label_array = []
        label_pos = np.where(volume_array == label_id)

        if len(label_pos[0]) == 0:
            return np.array(label_array)

        label_array = np.zeros((len(label_pos[0]), 3), np.float32)
        label_array[:, 0] = -label_pos[2] * spacing[0]
        label_array[:, 1] = -label_pos[1] * spacing[1]
        label_array[:, 2] = label_pos[0] * spacing[2]

        label_array = label_array + [volume_array.shape[2] * spacing[0], volume_array.shape[1] * spacing[1], 0]

        return label_array

    @staticmethod
    def resample_volumeImage(
            volume_image,
            new_spacing=1.0,
            isMask=False):

        resample_filter = sitk.ResampleImageFilter()

        if isMask is True:
            resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample_filter.SetInterpolator(sitk.sitkLinear)

        resample_filter.SetOutputDirection(volume_image.GetDirection())
        resample_filter.SetOutputOrigin(volume_image.GetOrigin())

        current_spacing = volume_image.GetSpacing()
        current_size = volume_image.GetSize()

        new_size = (int(current_size[0] * current_spacing[0] / new_spacing),
                    int(current_size[1] * current_spacing[1] / new_spacing),
                    int(current_size[2] * current_spacing[2] / new_spacing))

        spacing = (new_spacing, new_spacing, new_spacing)

        resample_filter.SetSize(new_size)
        resample_filter.SetOutputSpacing(spacing)
        new_volume_image = resample_filter.Execute(volume_image)

        return new_volume_image

    @staticmethod
    def generate_voxelgrid_from_pointcloud(points_src: torch.tensor, voxel_size=1):
        translate_distance = torch.zeros(3, dtype=torch.float32)
        translate_distance[0] = torch.min(points_src[:, 0]) - voxel_size
        translate_distance[1] = torch.min(points_src[:, 1]) - voxel_size
        translate_distance[2] = torch.min(points_src[:, 2]) - voxel_size

        points_src = points_src - translate_distance

        grid_shape = torch.zeros(3, dtype=torch.float32)
        grid_shape[0] = torch.max(points_src[:, 0]) / voxel_size + 2
        grid_shape[1] = torch.max(points_src[:, 1]) / voxel_size + 2
        grid_shape[2] = torch.max(points_src[:, 2]) / voxel_size + 2

        dims = torch.ceil(grid_shape).int()

        voxel_grid = torch.zeros([dims[0], dims[1], dims[2]], dtype=torch.uint8)

        points = torch.floor(points_src / voxel_size).int()

        voxel_grid[points[:, 0], points[:, 1], points[:, 2]] = 1

        return voxel_grid, translate_distance























