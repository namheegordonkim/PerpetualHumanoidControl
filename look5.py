from argparse import ArgumentParser

import imageio
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from tensorboardX import SummaryWriter
from tqdm import tqdm
from vispy import scene

from mlexp_utils import my_logging
from mlexp_utils.dirs import proj_dir
import numpy as np
import matplotlib.pyplot as plt

from viz.visual_data import BeatSaberVisualDataContainer, VisualDataContainer
from viz.visual_data_pv import XMLVisualDataContainer
from xror.xror import XROR
import vispy
import pyvista as pv

vispy.use("egl")


def main(args, remaining_args):
    if args.debug_yes:
        import pydevd_pycharm

        pydevd_pycharm.settrace(
            "localhost",
            port=12346,
            stdoutToServer=True,
            stderrToServer=True,
            suspend=False,
        )

    # For Tensorboard logging
    writer = SummaryWriter(log_dir=f"{proj_dir}/logdir/{args.run_name}/{args.out_name}")
    writer.add_text("args", str(args.__dict__))
    writer.add_text("remaining_args", str(remaining_args))

    logger = my_logging.get_logger(f"{args.out_name}")
    logger.info(f"Starting")

    visual_data = XMLVisualDataContainer("phc/data/assets/mjcf/smpl_humanoid.xml")

    posrot = torch.load("data/ours/posrot.pkl")
    rb_pos = posrot["rb_pos"]
    rb_rot = posrot["rb_rot"]

    pl = pv.Plotter(off_screen=False, window_size=(600, 600))
    pl.add_axes()
    plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=3, j_size=3)
    pl.add_mesh(plane)
    capsule = pv.Capsule(center=(0, 0, 0), direction=(1, 1, 1), radius=0.1)
    pl.add_mesh(capsule)
    pl.enable_shadows()
    pl.show()

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument(
        "--debug_yes", "-d", action="store_true"
    )  # if set, will pause the program
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
