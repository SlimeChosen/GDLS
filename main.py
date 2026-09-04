import src.config as config
from src.optimization import Optimizer

import warnings
warnings.simplefilter("ignore")


if __name__ == "__main__":
    cfg_path = 'config/liver.yaml'
    data_path = 'data/CHAOS/1.nii.gz'
    save_path = 'results/output.ply'
    label_id = 1

    cfg = config.load_config(cfg_path)

    optimizer = Optimizer(cfg_path, label_id)

    voxel_grid, translate_distance, spacing, gradient_grid = optimizer.preprocess(data_path)

    optimizer.run_gdls(voxel_grid, translate_distance, spacing, gradient_grid)

    optimizer.deform_mesh(data_path)

    optimizer.save_mesh_deformed(save_path)
