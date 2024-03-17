import os.path
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
from mlexp_utils.dirs import proj_dir
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

    # For Tensorboard logging
    # writer = SummaryWriter(log_dir=f"{proj_dir}/logdir/{args.run_name}/{args.out_name}")
    # writer.add_text("args", str(args.__dict__))
    # writer.add_text("remaining_args", str(remaining_args))
    d = torch.load(args.in_posrot_path)
    rr.init(d["exp_name"], recording_id=d["exp_name"])

    logger = my_logging.get_logger(f"{args.out_name}")
    logger.info(f"Starting")

    visual_data = XMLVisualDataContainer("phc/data/assets/mjcf/my_smpl_humanoid.xml")

    posrot = torch.load(args.in_posrot_path)
    rb_pos = posrot["rb_pos"]
    rb_rot = posrot["rb_rot"]
    cumulative_r = np.mean(posrot["cumulative_rewards"], axis=(-1)).reshape(-1)

    rr.set_time_sequence("epoch", d["epoch"])
    rr.log(f"CumulativeReward", rr.Scalar(cumulative_r))

    # my_3p = torch.load(args.in_3p_path)
    # ref_rb_pos_subset = my_3p["ref_rb_pos_subset"][None].detach().cpu().numpy()
    ref_rb_pos = posrot["ref_rb_pos"]

    pl = pv.Plotter(off_screen=True, window_size=(608, 608))
    # pl = pv.Plotter(off_screen=False, window_size=(608, 608))
    actors = []
    ax_actors = []
    pl.add_mesh(visual_data.plane)
    for mesh, ax in zip(visual_data.meshes, visual_data.axes):
        actor = pl.add_mesh(mesh, color="green")
        actors.append(actor)

    target_actors = []
    for i in range(24):
        actor = pl.add_mesh(visual_data.targets[i], color="red")
        target_actors.append(actor)

    pl.enable_shadows()

    imgs = []
    for t in tqdm(range(0, rb_pos.shape[1], 4)):
    # for t in tqdm(range(0, 12, 2)):
        for i, actor in enumerate(actors):
            # ax_actor = ax_actors[i]
            m = np.eye(4)
            pos = rb_pos[0, t, i] * 1
            quat = rb_rot[0, t, i] * 1
            m[:3, :3] = Rotation.from_quat(quat).as_matrix()
            m[:3, 3] = pos
            actor.user_matrix = m
            # ax_actor.user_matrix = m

        for i, actor in enumerate(target_actors):
            m = np.eye(4)
            pos = ref_rb_pos[0, t, i] * 1
            m[:3, 3] = pos
            actor.user_matrix = m

        distance = 5
        pl.camera.position = (distance, distance, 2)
        pl.camera.focal_point = (0, 0, 0)
        pl.render()
        # pl.show()

        # plt.figure()
        img = np.array(pl.screenshot())
        # img = pl.screenshot()
        # plt.imshow(img)
        # plt.show()
        imgs.append(img)

    for i, img in enumerate(imgs):
        # rr.set_time_sequence("epoch", d['epoch'])
        rr.set_time_sequence("frame", i)
        rr.log(f"EvalVideo/{d['epoch']:06d}", rr.Image(img))
    plt.close()
    ddd = f"{proj_dir}/logdir/{d['out_name']}"
    os.makedirs(ddd, exist_ok=True)
    rr.save(f"{ddd}/epoch{d['epoch']:06d}.rrd")
    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--in_posrot_path", type=str, required=True)
    # parser.add_argument("--in_3p_path", type=str, required=True)
    # parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument(
        "--debug_yes", "-d", action="store_true"
    )  # if set, will pause the program
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
