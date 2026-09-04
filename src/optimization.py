import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from tqdm import tqdm

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency)

from src.volumeImage import VolumeImage
from src.GDLS import GDLS
import src.config as config


class Optimizer():
    def __init__(self, cfg, label_id):
        self.cfg = None
        self.label_id = label_id

        self.src_points = None
        self.src_normals = None

        self.dilate = None
        self.dilate_scale = 1.0

        if isinstance(cfg, str):
            self.cfg = config.load_config(cfg)
        if isinstance(cfg, dict):
            self.cfg = cfg

        self.sample_num = self.cfg['optim']['sample_num']

        self.center = None
        self.scale = None

        self.mesh_src = None
        self.mesh_deformed = None

    def set_weights(self, w_chamfer, w_edge, w_normal, w_laplacian):
        self.cfg['optim']['w_chamfer'] = w_chamfer
        self.cfg['optim']['w_edge'] = w_edge
        self.cfg['optim']['w_normal'] = w_normal
        self.cfg['optim']['w_laplacian'] = w_laplacian

    def preprocess(self, data_path):
        voxel_size = self.cfg['preprocessing']['voxel_size']
        resample_size = self.cfg['preprocessing']['resample_size']

        self.dilate = self.cfg['preprocessing']['dilate']

        voxel_grid, translate_distance, spacing = VolumeImage.get_voxel_grid_from_mask(
            data_path, self.label_id, voxel_size, resample_size)

        if self.dilate is True:
            voxel_grid, self.dilate_scale = Optimizer.dilate_voxel_grid(voxel_grid)

        gradient_grid = VolumeImage.compute_voxel_grid_gradient(voxel_grid)
        gradient_grid = torch.stack((gradient_grid[0], gradient_grid[1], gradient_grid[2]), dim=-1)
        gradient_grid = gradient_grid / gradient_grid.norm(p=2, dim=-1, keepdim=True)

        voxel_grid_edge = VolumeImage.get_voxel_grid_edge(voxel_grid)

        return voxel_grid_edge, translate_distance, spacing, gradient_grid

    def run_gdls(self, voxel_grid, translate_distance, spacing, gradient_grid):
        gdls = GDLS(self.cfg)
        # dls.gradient_grid = -gradient_grid

        gdls.set_voxel_grid(voxel_grid, gradient_grid)

        gdls.run_gdls()

        refined_points = gdls.refine_voxel_points()
        self.src_points = GDLS.get_world_coor_points(refined_points, gdls.voxel_size, translate_distance, spacing)
        self.src_normals = gdls.normals

        invalid_point = torch.isnan(self.src_normals) | torch.isinf(self.src_normals)
        mask = ~invalid_point.any(dim=1)
        self.src_points = self.src_points[mask]
        self.src_normals = self.src_normals[mask]

        if self.dilate is True:
            diff = torch.tensor([gdls.voxel_size, gdls.voxel_size, gdls.voxel_size], dtype=torch.float32)
            self.src_points = Optimizer.refine_dilate_points(self.dilate_scale, self.src_points - diff, self.src_normals)

    def init_tar_verts(self, data_path):
        voxel_grid, translate_distance, spacing = VolumeImage.get_voxel_grid_edge_from_mask(
            data_path, self.label_id, 1, 1)

        verts = GDLS.get_pointcloud_from_voxelgrid(voxel_grid, 1)
        verts = GDLS.get_world_coor_points(verts, 1, translate_distance, spacing)

        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale

        if len(verts) < self.sample_num:
            self.sample_num = len(verts)

        self.center = center
        self.scale = scale

        return verts, center, scale

    def init_src_mesh(self, psr_depth):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(self.src_points))
        pcd.normals = o3d.utility.Vector3dVector(np.array(self.src_normals))

        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=psr_depth)

        clusters, clusters_n, _ = mesh.cluster_connected_triangles()
        clusters = np.asarray(clusters)
        clusters_n = np.asarray(clusters_n)
        largest_idx = clusters_n.argmax()
        remove_mask = clusters != largest_idx
        mesh.remove_triangles_by_mask(remove_mask)
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_non_manifold_edges()

        verts = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.triangles, dtype=torch.int64)

        self.mesh_src = Meshes(verts=[verts], faces=[faces])

        return self.mesh_src

    def deform_mesh(self, data_path, psr_depth=9):
        Niter = self.cfg['optim']['Niter']
        w_chamfer = self.cfg['optim']['w_chamfer']
        w_edge = self.cfg['optim']['w_edge']
        w_normal = self.cfg['optim']['w_normal']
        w_laplacian = self.cfg['optim']['w_laplacian']

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")

        tar_verts, center, scale = self.init_tar_verts(data_path)
        tar_verts = tar_verts.to(device)

        if self.mesh_src is None:
            src_mesh = self.init_src_mesh(psr_depth)
        else:
            src_mesh = self.mesh_src

        verts, faces = src_mesh.get_mesh_verts_faces(0)
        verts = (verts - center) / scale
        src_mesh = Meshes(verts=[verts], faces=[faces]).to(device)

        deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
        optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

        # loop = range(Niter)
        loop = tqdm(range(Niter), desc='Deforming mesh')

        for i in loop:
            optimizer.zero_grad()

            new_src_mesh = src_mesh.offset_verts(deform_verts)

            random_idx = torch.randint(0, len(tar_verts), (self.sample_num,))
            sorted_idx = torch.sort(random_idx).values
            sample_trg = torch.zeros([1, self.sample_num, 3], dtype=torch.float32)
            sample_trg[0] = tar_verts[sorted_idx]
            sample_trg = sample_trg.to(device)
            sample_src = sample_points_from_meshes(new_src_mesh, self.sample_num)

            loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

            loss_edge = mesh_edge_loss(new_src_mesh)

            loss_normal = mesh_normal_consistency(new_src_mesh)

            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

            loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian

            loss.backward()
            optimizer.step()

            final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

        final_verts = final_verts.to('cpu') * scale + center

        self.mesh_deformed = Meshes(verts=[final_verts.to('cpu')], faces=[final_faces.to('cpu')])

    def save_mesh_src(self, save_path):
        if self.mesh_src is None:
            assert 'mesh_src is None'
        verts, faces = self.mesh_src.get_mesh_verts_faces(0)
        verts = verts * self.scale + self.center

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(verts))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))

        o3d.io.write_triangle_mesh(save_path, mesh)

    def save_mesh_deformed(self, save_path):
        if self.mesh_deformed is None:
            assert 'mesh_deformed is None'
        verts, faces = self.mesh_deformed.get_mesh_verts_faces(0)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(faces.detach().cpu().numpy())

        o3d.io.write_triangle_mesh(save_path, mesh)

    @staticmethod
    def dilate_voxel_grid(voxel_grid: torch.tensor):
        point_pos = torch.where(voxel_grid > 0)
        height = point_pos[2].max() - point_pos[2].min()
        scale = float(height + 2) / height

        shape = voxel_grid.shape
        voxel_conv = torch.zeros(1, 1, shape[0], shape[1], shape[2])
        voxel_conv[0][0] = voxel_grid

        kernel = torch.ones(1, 1, 3, 3, 3)

        out = F.conv3d(voxel_conv, kernel, padding=2)
        out = out.clamp(min=0)
        out = out[0][0]
        out[out > 0] = 1

        return out, scale

    @staticmethod
    def refine_dilate_points(dilate_scale, points_src:torch.tensor, normals_src:torch.tensor):
        center = points_src.mean(0)
        new_points = points_src - center
        new_points = new_points - dilate_scale * normals_src
        new_points = new_points + center
        return new_points















