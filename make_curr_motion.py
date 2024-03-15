import imageio
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from utils.motion_lib_base import compute_motion_dof_vels
from argparse import ArgumentParser

from mlexp_utils import my_logging
from mlexp_utils.dirs import super_dir
from poselib.poselib.skeleton.skeleton3d import (
    SkeletonState,
    SkeletonTree,
    SkeletonMotion,
)
import pyvista as pv
import torch

from viz.visual_data_pv import XMLVisualDataContainer


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

    # correction_rot = Rotation.from_euler("ZYX", [-0.5 * np.pi, -0.5 * np.pi, 0])
    correction_rot = Rotation.from_euler("Y", 0.5 * np.pi)
    correction_matrix = correction_rot.as_matrix()

    d = torch.load(args.in_path)
    hehe = Rotation.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
    pose_quat_global = d["pose_quat_global"]
    original_root_quat = pose_quat_global[0, 0] * 1
    original_root_rot = Rotation.from_quat(original_root_quat) * hehe
    m = Rotation.from_quat(pose_quat_global.reshape(-1, 4))
    m = m * hehe
    # m2 = correction_rot * m

    pose_quat_global = m.as_quat().reshape(pose_quat_global.shape)
    # pose_quat_global2 = m2.as_quat().reshape(pose_quat_global.shape)

    # pose_quat_global[:, 0] = pose_quat_global2[:, 0]
    # pose_quat_global = torch.as_tensor(pose_quat_global)
    # pose_quat_global = torch.as_tensor(pose_quat_global2)

    # pose_quat_global[..., [0, 1, 2, 3]] = pose_quat_global[..., [0, 2, 1, 3]]
    pose_quat_global[..., [0, 1, 2, 3]] = pose_quat_global[..., [1, 0, 2, 3]]
    pose_quat_global[..., 0] *= -1

    m = Rotation.from_quat(pose_quat_global.reshape(-1, 4))
    m = correction_rot * m
    pose_quat_global = m.as_quat().reshape(pose_quat_global.shape)
    pose_quat_global = torch.as_tensor(pose_quat_global)

    trans = d["trans"]
    trans[..., [0, 1, 2]] = trans[..., [2, 0, 1]]
    trans[..., [0, 1]] -= trans[0, [0, 1]]
    trans[..., 2] -= 0.15
    trans = torch.as_tensor(trans)

    sk_tree = SkeletonTree.from_mjcf("phc/data/assets/mjcf/smpl_humanoid_1.xml")
    sk_state = SkeletonState.from_rotation_and_root_translation(
        sk_tree, pose_quat_global, trans, is_local=False
    )
    # sk_state = SkeletonState.from_rotation_and_root_translation(
    #     sk_tree, pose_quat_local, trans, is_local=True
    # )
    curr_motion = SkeletonMotion.from_skeleton_state(sk_state, 60)
    curr_dof_vels = compute_motion_dof_vels(curr_motion)
    curr_motion.curr_dof_vels = curr_dof_vels

    torch.save(curr_motion, args.out_path)

    rb_pos = curr_motion.global_translation.cpu().numpy()[None]
    rb_rot = curr_motion.global_rotation.cpu().numpy()[None]

    visual_data = XMLVisualDataContainer("phc/data/assets/mjcf/my_smpl_humanoid.xml")
    pl = pv.Plotter(off_screen=True, window_size=(608, 608))
    # pl = pv.Plotter(off_screen=False, window_size=(608, 608))
    actors = []
    pl.add_mesh(visual_data.plane)
    pl.add_axes()
    for mesh, ax in zip(visual_data.meshes, visual_data.axes):
        actor = pl.add_mesh(mesh, color="red")
        actors.append(actor)

    target_actors = []
    for i in range(24):
        actor = pl.add_mesh(visual_data.targets[i], color="red")
        target_actors.append(actor)

    pl.enable_shadows()

    imgs = []
    # for t in tqdm(range(0, 400, 4)):
    for t in tqdm(range(1360, 1600, 4)):
        # for t in tqdm(range(0, rb_pos.shape[1], 4)):
        for i, actor in enumerate(actors):
            m = np.eye(4)
            pos = rb_pos[0, t, i] * 1
            quat = rb_rot[0, t, i] * 1
            m[:3, :3] = Rotation.from_quat(quat).as_matrix()
            m[:3, 3] = pos
            actor.user_matrix = m
            # ax_actor.user_matrix = m

        distance = 5
        pl.camera.position = (-distance, -distance, 4)
        pl.camera.focal_point = (0, 0, 0)
        # pl.camera.up = (0, 0, 1)
        pl.render()
        # pl.show()

        # plt.figure()
        # img = np.array(pl.screenshot())
        img = pl.screenshot()
        # plt.imshow(img)
        # plt.show()
        imgs.append(img)

    w = imageio.get_writer(
        f"{super_dir}/dump/movie.mp4",
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
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument(
        "--debug_yes", "-d", action="store_true"
    )  # if set, will pause the program
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
