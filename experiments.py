import os.path
import os
import numpy as np

from src.optimization import Optimizer
import pyvista as pv
import datetime
from src import config

import warnings
warnings.simplefilter("ignore")

def draw_voxel(voxel_grid, save_path, off_screen=True, pic_name=''):
    voxel_grid = voxel_grid.numpy()

    grid = pv.ImageData()
    grid.dimensions = np.array(voxel_grid.shape) + 1
    grid.origin = (0,0,0)
    grid.spacing = (1,1,1)

    grid.cell_data['values'] = voxel_grid.flatten(order='F')
    threshed = grid.threshold([0.9, 1.1], scalars='values')

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_mesh(threshed, color='#808080')
    plotter.view_vector([1, 1.5, 1])
    plotter.show(screenshot=save_path + '/' + pic_name + "voxel_view_0.png")

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_mesh(threshed, color='#808080')
    plotter.view_vector([-1, 1.5, 1])
    plotter.show(screenshot=save_path + '/' + pic_name + "voxel_view_1.png")

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_mesh(threshed, color='#808080')
    plotter.view_vector([1, -1.5, 1])
    plotter.show(screenshot=save_path + '/' + pic_name + "voxel_view_2.png")

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_mesh(threshed, color='#808080')
    plotter.view_vector([-1, -1.5, 1])
    plotter.show(screenshot=save_path + '/' + pic_name + "voxel_view_3.png")

def draw_1x1_voxel(cfg, result_path, off_screen=True):
    label_id = cfg['preprocessing']['label_id']
    nii_path = cfg['config']['data']

    save_path = result_path + "/label_" + str(cfg['preprocessing']['label_id'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    from src.volumeImage import VolumeImage
    voxel_grid, translate_distance, spacing = VolumeImage.get_voxel_grid_from_mask(
        nii_path, label_id, 1, 1)

    draw_voxel(voxel_grid, save_path, off_screen, 'full_')

def run_optim(cfg, result_path):
    save_path = result_path + "/label_" + str(cfg['preprocessing']['label_id'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    mesher = Optimizer(cfg)

    t0_total = datetime.datetime.now()

    print('preprocessing: ', end='')
    t0 = datetime.datetime.now()
    voxel_grid, translate_distance, spacing, gradient_grid = mesher.preprocess()
    t1 = datetime.datetime.now()
    print(t1 - t0)
    draw_voxel(voxel_grid, save_path)

    print("diffusion least square:")
    mesher.run_gdls(voxel_grid, translate_distance, spacing, gradient_grid)

    print("deforming mesh:")
    _ = mesher.deform_mesh()

    mesher.save_mesh_src(save_path + "/mesh_dls_src.ply")
    mesher.save_mesh_deformed(save_path + "/mesh_dls_deformed.ply")

    t1_total = datetime.datetime.now()
    print("total time used: " + str(t1_total - t0_total))

def run_PSR(cfg, result_path):
    from model.PSR import PSR
    save_path = result_path + "/label_" + str(cfg['preprocessing']['label_id'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    psr = PSR(cfg)

    t0_total = datetime.datetime.now()

    # print('preprocessing: ', end='')
    # t0 = datetime.datetime.now()
    voxel_grid_edge, translate_distance, spacing, gradient_grid = psr.preprocess()
    # t1 = datetime.datetime.now()
    # print(t1 - t0)

    # print('surface_reconstruction: ', end='')
    # t0 = datetime.datetime.now()
    psr.surface_reconstruction(voxel_grid_edge, translate_distance, spacing, gradient_grid)
    # psr.save_mesh_src(save_path + "/mesh_psr_src.ply")
    # t1 = datetime.datetime.now()
    # print(t1 - t0)

    psr.mesh_src_laplacian(300, 0.1)
    psr.save_mesh_laplacian(save_path + "/mesh_psr_laplacian.ply")

    # psr.mesh_src_taubin(100)
    # psr.save_mesh_taubin(save_path + "/mesh_psr_taubin.ply")

    # psr.deform_mesh(psr.mesh_taubin)
    # psr.save_mesh_deformed(save_path + "/mesh_psr_taubin_deformed.ply")

    print('PSR total: ', end='')
    t1_total = datetime.datetime.now()
    print(t1_total - t0_total)

def run_MC(cfg, result_path):
    from model.MC import MC
    save_path = result_path + "/label_" + str(cfg['preprocessing']['label_id'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    mc = MC(cfg)

    t0_total = datetime.datetime.now()

    # print('preprocessing: ', end='')
    # t0 = datetime.datetime.now()
    voxel_grid, translate_distance, spacing= mc.preprocess()
    # t1 = datetime.datetime.now()
    # print(t1 - t0)

    print('MC surface_reconstruction: ', end='')
    t0 = datetime.datetime.now()
    mc.surface_reconstruction(voxel_grid, translate_distance)
    # mc.save_mesh_src(save_path + "/mesh_mc_src.ply")
    t1 = datetime.datetime.now()
    print(t1 - t0)

    mc.mesh_src_laplacian(300, 0.1)
    mc.save_mesh_laplacian(save_path + "/mesh_mc_laplacian.ply")

    # mc.mesh_src_taubin(100)
    # mc.save_mesh_taubin(save_path + "/mesh_psr_taubin.ply")
    #
    # mc.deform_mesh(mc.mesh_taubin)
    # mc.save_mesh_deformed(save_path + "/mesh_psr_taubin_deformed.ply")

    print('MC total: ', end='')
    t1_total = datetime.datetime.now()
    print(t1_total - t0_total)

def run_meshFit_sphere(cfg_path, result_path):
    from model.meshFit_sphere import MeshFit
    cfg = config.load_config(cfg_path)
    save_path = result_path + "/label_" + str(cfg['preprocessing']['label_id'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    meshfit = MeshFit(cfg_path)

    print('preprocessing: ', end='')
    t0 = datetime.datetime.now()
    voxel_grid, translate_distance, spacing= meshfit.preprocess()
    t1 = datetime.datetime.now()
    print(t1 - t0)

    print('surface_reconstruction: ', end='')
    t0 = datetime.datetime.now()
    meshfit.surface_reconstruction(voxel_grid)
    t1 = datetime.datetime.now()
    print(t1 - t0)

    meshfit.deform_mesh()
    meshfit.save_mesh_deformed(save_path + "/mesh_sphere_deformed.ply")

def run_multi_jobs():
    result_path = "result/experiments"

    cfg = config.load_config("config/liver.yaml")
    start_num = 1
    end_num = 21

    for i in range(start_num, end_num):
        cfg['config']['data'] = "/home/lemmon/Project/pneumoperitoneum/segment/word_" + str(i).rjust(4,
                                                                                                     '0') + "_total.nii.gz"

        word_path = result_path + "/word_" + str(i).rjust(3, '0')
        if not os.path.exists(word_path):
            os.mkdir(word_path)

        run_PSR(cfg, word_path)
        run_MC(cfg, word_path)
        # draw_1x1_voxel(cfg, word_path, True)

    # for i in range(start_num, end_num):
    #     cfg['config']['data'] = "/home/lemmon/Project/pneumoperitoneum/segment/word_" + str(i).rjust(4, '0') + "_total.nii.gz"
    #     word_path = result_path + "/word_" + str(i).rjust(3, '0')
    #     if not os.path.exists(word_path):
    #         os.mkdir(word_path)
    #     run_optim(cfg, word_path)

    cfg = config.load_config("config/ilium.yaml")
    for i in range(start_num, end_num):
        cfg['config']['data'] = "/home/lemmon/Project/pneumoperitoneum/segment/word_" + str(i).rjust(4,
                                                                                                     '0') + "_total.nii.gz"

        word_path = result_path + "/word_" + str(i).rjust(3, '0')
        if not os.path.exists(word_path):
            os.mkdir(word_path)

        run_PSR(cfg, word_path)
        run_MC(cfg, word_path)
    #     draw_1x1_voxel(cfg, word_path, True)

    # for i in range(start_num, end_num):
    #     cfg['config']['data'] = "/home/lemmon/Project/pneumoperitoneum/segment/word_" + str(i).rjust(4,
    #                                                                                                  '0') + "_total.nii.gz"
    #     word_path = result_path + "/word_" + str(i).rjust(3, '0')
    #     if not os.path.exists(word_path):
    #         os.mkdir(word_path)
    #
    #     run_optim(cfg, word_path)

    cfg = config.load_config("config/lumbarSpine.yaml")
    for i in range(start_num, end_num):
        cfg['config']['data'] = "/home/lemmon/Project/pneumoperitoneum/segment/word_" + str(i).rjust(4,
                                                                                                     '0') + "_total.nii.gz"

        word_path = result_path + "/word_" + str(i).rjust(3, '0')
        if not os.path.exists(word_path):
            os.mkdir(word_path)

        run_PSR(cfg, word_path)
        run_MC(cfg, word_path)
        # draw_1x1_voxel(cfg, word_path, True)

    # for i in range(start_num, end_num):
    #     cfg['config']['data'] = "/home/lemmon/Project/pneumoperitoneum/segment/word_" + str(i).rjust(4,
    #                                                                                                  '0') + "_total.nii.gz"
    #     word_path = result_path + "/word_" + str(i).rjust(3, '0')
    #     if not os.path.exists(word_path):
    #         os.mkdir(word_path)
    #
    #     run_optim(cfg, word_path)


if __name__ == "__main__":
    # result_path = "result/test"

    # cfg = config.load_config("config/liver.yaml")
    # run_MC(cfg, result_path)
    # run_PSR(cfg, result_path)
    # run_optim(cfg, result_path)

    # cfg = config.load_config("config/ilium.yaml")
    # run_MC(cfg, result_path)
    # run_PSR(cfg, result_path)
    # run_optim(cfg, result_path)
    #
    # cfg = config.load_config("config/lumbarSpine.yaml")
    # run_MC(cfg, result_path)
    # run_PSR(cfg, result_path)
    # run_optim(cfg, result_path)

    cfg = config.load_config("config/chaos.yaml")

    test_idx_list = [1, 14, 23, 26, 30]

    result_path = "/home/lemmon/Project/DQS/result/chaos/mc"
    run_MC(cfg, result_path)

    result_path = "/home/lemmon/Project/DQS/result/chaos/psr"
    run_PSR(cfg, result_path)

