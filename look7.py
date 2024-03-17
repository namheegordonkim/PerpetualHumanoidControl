from argparse import ArgumentParser

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import vispy
from scipy.spatial.transform import Rotation
from tensorboardX import SummaryWriter
from tqdm import tqdm

import pyvista as pv
from mlexp_utils import my_logging
from mlexp_utils.dirs import proj_dir, proj_dir
from viz.visual_data_pv import XMLVisualDataContainer
import rerun as rr


def main(args, remaining_args):
    if args.debug_yes:
        import pydevd_pycharm

        pydevd_pycharm.settrace(
            "localhost",
            port=12345,
            stdoutToServer=True,
            stderrToServer=True,
            suspend=False,
        )

    logger = my_logging.get_logger(f"{args.out_name}")
    logger.info(f"Starting")

    curr_motion = torch.load(args.in_path)

    rb_pos = curr_motion.global_translation.cpu().numpy()[None]
    rb_rot = curr_motion.global_rotation.cpu().numpy()[None]

    visual_data = XMLVisualDataContainer("phc/data/assets/mjcf/my_smpl_humanoid.xml")
    pl = pv.Plotter(off_screen=True, window_size=(608, 608))
    # pl = pv.Plotter(off_screen=False, window_size=(608, 608))
    actors = []
    pl.add_mesh(visual_data.plane)
    for mesh, ax in zip(visual_data.meshes, visual_data.axes):
        actor = pl.add_mesh(mesh, color="red")
        actors.append(actor)

    target_actors = []
    for i in range(24):
        actor = pl.add_mesh(visual_data.targets[i], color="red")
        target_actors.append(actor)

    pl.enable_shadows()

    imgs = []
    for t in tqdm(range(0, rb_pos.shape[1], 4)):
        for i, actor in enumerate(actors):
            m = np.eye(4)
            pos = rb_pos[0, t, i] * 1
            quat = rb_rot[0, t, i] * 1
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
        img = np.array(pl.screenshot())
        img = pl.screenshot()
        # plt.imshow(img)
        # plt.show()
        imgs.append(img)

    w = imageio.get_writer(
        f"{proj_dir}/dump/movie.mp4",
        format="FFMPEG",
        mode="I",
        fps=15,
        codec="h264",
        pixelformat="yuv420p",
    )
    for img in imgs:
        w.append_data(img)
    w.close()

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--in_path", type=str, required=True)
    parser.add_argument(
        "--debug_yes", "-d", action="store_true"
    )  # if set, will pause the program
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
