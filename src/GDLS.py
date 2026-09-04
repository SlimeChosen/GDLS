import torch
import multiprocessing
import torch_geometric

from tqdm import tqdm

import src.utils as methods


neighbors_offsets = torch.tensor([[-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
                                 [-1,  0, -1], [-1,  0,  0], [-1,  0,  1],
                                 [-1,  1, -1], [-1,  1,  0], [-1,  1,  1],
                                 [ 0, -1, -1], [ 0, -1,  0], [ 0, -1,  1],
                                 [ 0,  0, -1], [ 0,  0,  1],
                                 [ 0,  1, -1], [ 0,  1,  0], [ 0,  1,  1],
                                 [ 1, -1, -1], [ 1, -1,  0], [ 1, -1,  1],
                                 [ 1,  0, -1], [ 1,  0,  0], [ 1,  0,  1],
                                 [ 1,  1, -1], [ 1,  1,  0], [ 1,  1,  1]], dtype=torch.int)

class GDLS(object):

    def __init__(self, cfg):
        self.voxel_grid_edge = None
        self.voxel_points = None
        self.features = None
        self.normals = None
        self.conv_info = None
        self.gradient_grid = None

        self.search_graph = None

        self.voxel_points_dict = None

        self.voxel_size = cfg['preprocessing']['voxel_size']
        self.max_conv_times = cfg['gdls']['max_conv_times']
        self.max_mse = cfg['gdls']['max_mse']

    def set_voxel_grid(self, voxel_grid_edge:torch.tensor, gradient_grid:torch.tensor):
        self.gradient_grid = -gradient_grid
        self.voxel_grid_edge = voxel_grid_edge

        self.voxel_points = GDLS.sample_from_voxel_grid(voxel_grid_edge)


        self.features = torch.zeros([len(self.voxel_points), 6], dtype=torch.float32)
        self.normals = torch.zeros([len(self.voxel_points), 3], dtype=torch.float32)

        self.conv_info = torch.zeros([len(self.voxel_points), 2], dtype=torch.float32)

        self.voxel_points_dict = {}
        for idx, point in enumerate(self.voxel_points):
            p = (int(point[0]), int(point[1]), int(point[2]))
            self.voxel_points_dict[p] = idx

        self.search_graph = self.build_search_graph()

    def build_search_graph(self):
        pool = multiprocessing.Pool()
        pool_result = pool.map(self.build_search_graph_in_cpu, self.voxel_points)
        pool.close()
        pool.join()

        result_list = [item for sublist in pool_result for item in sublist]

        edges = torch.tensor(result_list, dtype=torch.long).t().contiguous()

        graph = torch_geometric.data.Data(x=self.voxel_points, edge_index=edges)

        return graph

    def build_search_graph_in_cpu(self, center_point):
        edge = []

        p_c = (int(center_point[0]), int(center_point[1]), int(center_point[2]))
        center_point_idx = self.voxel_points_dict[p_c]

        for offset in neighbors_offsets:
            p = center_point + offset
            if self.voxel_grid_edge[p[0], p[1], p[2]] > 0:
                p_dict = (int(p[0]), int(p[1]), int(p[2]))
                p_idx = self.voxel_points_dict[p_dict]
                edge.append([center_point_idx, p_idx])

        return edge

    def run_gdls(self):
        for i in tqdm(range(len(self.voxel_points)), desc='GDLS'):
            self.normals[i], self.features[i], self.conv_info[i] = self.gdls_in_cpu(self.voxel_points[i])

    def gdls_in_cpu(self, center_point):
        features = torch.zeros([1, 6], dtype=torch.float32)
        conv_info = torch.zeros([1, 2], dtype=torch.float32)

        last_mse = 10000

        normal = self.gradient_grid[center_point[0], center_point[1], center_point[2]].flatten()

        if (torch.any(torch.isnan(normal)) or torch.any(torch.isinf(normal))):
            points_ = self.get_graph_neighbor(center_point, 2)
            normal_ = torch.zeros(3, dtype=torch.float32)
            cnt_ = 0
            for p_ in points_:
                n_ = self.gradient_grid[int(p_[0]), int(p_[1]), int(p_[2])].flatten()
                if (torch.any(torch.isnan(normal)) or torch.any(torch.isinf(normal))):
                    continue
                normal_ += n_
                cnt_ += 1
            normal = normal_ / cnt_

        if torch.any(torch.isnan(normal)) or torch.any(torch.isinf(normal)):
            return normal, features, conv_info

        for conv_stage in range(2, self.max_conv_times+1):
            points = self.get_graph_neighbor(center_point, conv_stage)
            points = GDLS.remove_invalid_points(points)

            points, R = methods.rotate_points_z_axis(points - center_point, normal)

            fit_result, mse, total_error = methods.fit_quadric_surface(points)

            mes_rate = (mse - last_mse) / last_mse
            if conv_stage <= 3:
                features = torch.tensor([
                    fit_result[0], fit_result[1], fit_result[2],
                    fit_result[3], fit_result[4], fit_result[5]
                ], dtype=torch.float32)
                conv_info = torch.tensor([conv_stage, mse], dtype=torch.float32)
                last_mse = mse

            elif mes_rate <= 0.2 and mse < self.max_mse:
                features = torch.tensor([
                    fit_result[0], fit_result[1], fit_result[2],
                    fit_result[3], fit_result[4], fit_result[5]
                ], dtype=torch.float32)
                conv_info = torch.tensor([conv_stage, mse], dtype=torch.float32)
                last_mse = mse

            else:
                break

        return normal, features, conv_info

    def get_graph_neighbor(self, center_point, depth):
        p = (int(center_point[0]), int(center_point[1]), int(center_point[2]))
        node_idx = self.voxel_points_dict[p]

        sampler = torch_geometric.loader.NeighborLoader(
            self.search_graph, num_neighbors=[-1]*(depth-1), input_nodes=torch.tensor([node_idx]),
            batch_size=1, shuffle=False, directed=False)

        points = torch.tensor([], dtype=torch.float32)
        for batch in sampler:
            points = batch.x

        return points.float()

    def refine_voxel_points(self):
        pool = multiprocessing.Pool()
        pool_result = pool.map(self.refine_voxel_points_self_in_cpu, self.voxel_points)
        pool.close()
        pool.join()

        refined_points = torch.cat(pool_result, dim=0).view(len(pool_result), 3)

        return refined_points

    def refine_voxel_points_self_in_cpu(self, center_point):
        normal_z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        center_point_pos = self.voxel_points_dict[(int(center_point[0]), int(center_point[1]), int(center_point[2]))]

        normal = self.normals[center_point_pos]
        if torch.any(torch.isnan(normal)) or torch.any(torch.isinf(normal)):
            return torch.zeros([1, 3], dtype=torch.float32)

        sample_point = methods.sample_from_qudric(self.features[center_point_pos], torch.zeros([1, 3], dtype=torch.float32))
        sample_point, _ = methods.rotate_points(sample_point, normal_z, normal)
        sample_point = sample_point + center_point

        return sample_point

    @staticmethod
    def get_world_coor_points(points_src:torch.tensor, voxel_size, translate_distance:torch.tensor, spacing):
        points = points_src.clone().detach()

        points = points * voxel_size + translate_distance

        points[:, 0] = points[:, 0] * spacing[0]
        points[:, 1] = points[:, 1] * spacing[1]
        points[:, 2] = points[:, 2] * spacing[2]

        return points

    @staticmethod
    def sample_from_voxel_grid(voxel_grid:torch.tensor)->torch.tensor:
        voxel_pos = torch.where(voxel_grid > 0)
        voxel_points = torch.stack(
            [torch.tensor(voxel_pos[0]), torch.tensor(voxel_pos[1]), torch.tensor(voxel_pos[2])], dim=1)

        return voxel_points

    @staticmethod
    def get_pointcloud_from_voxelgrid(voxel_grid:torch.tensor, voxel_value:int):
        points_pos = torch.where((voxel_grid > 0) & (voxel_grid <= voxel_value))

        pcd = torch.stack([points_pos[0], points_pos[1], points_pos[2]], dim=1)

        return pcd.float()

    @staticmethod
    def remove_invalid_points(points):
        invalid_point = torch.isnan(points) | torch.isinf(points)

        mask = ~invalid_point.any(dim=1)

        points = points[mask]

        return points

















