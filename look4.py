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

    visual_data = XMLVisualDataContainer("phc/data/assets/mjcf/my_smpl_humanoid.xml")

    posrot = torch.load("data/ours/posrot.pkl")
    rb_pos = posrot["rb_pos"]
    rb_rot = posrot["rb_rot"]

    pl = pv.Plotter(off_screen=True, window_size=(608, 608))
    # pl = pv.Plotter(off_screen=False, window_size=(608, 608))
    actors = []
    ax_actors = []
    pl.add_mesh(visual_data.plane)
    for mesh, ax in zip(visual_data.meshes, visual_data.axes):
        actor = pl.add_mesh(mesh, color="green")
        actors.append(actor)
        # ax_actor = pl.add_mesh(ax)
        # ax_actors.append(ax_actor)
    pl.enable_shadows()

    imgs = []
    for t in range(0, rb_pos.shape[0]):
        for i, actor in enumerate(actors):
            # ax_actor = ax_actors[i]
            m = np.eye(4)
            pos = rb_pos[t, i] * 1
            quat = rb_rot[t, i] * 1
            m[:3, :3] = Rotation.from_quat(quat).as_matrix()
            m[:3, 3] = pos
            actor.user_matrix = m
            # ax_actor.user_matrix = m

        distance = 5
        pl.camera.position = (distance, distance, 2)
        pl.camera.focal_point = (0, 0, 0)
        pl.render()
        # pl.show()

        # plt.figure()
        img = pl.screenshot()
        # plt.imshow(img)
        # plt.show()
        imgs.append(img)

    w = imageio.get_writer(
        "dump/movie.mp4",
        format="FFMPEG",
        mode="I",
        fps=15,
        codec="h264",
        pixelformat="yuv420p",
    )
    for img in imgs:
        w.append_data(img)
    w.close()

    plt.close()

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
