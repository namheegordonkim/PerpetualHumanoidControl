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
from xror.xror import XROR
import vispy

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

    # filepath = f"{proj_dir}/dataset/025bcf95-9be6-4620-8d68-6b5903e62aee/163bd29f-3209-4890-936f-3fc77dccd19b.xror"
    # filepath = f"{proj_dir}/dataset/025bcf95-9be6-4620-8d68-6b5903e62aee/284e1d66-7bba-4160-969b-0e3ccd0dc5e6.xror"
    filepath = f"{proj_dir}/data/theirs/their_vr_input.pkl"
    d = torch.load(filepath, map_location="cpu")
    frames = torch.cat([d["ref_rb_pos_subset"], d["ref_rb_rot_subset"]], dim=-1)
    frames_np = np.array(frames[0])
    frames_np[..., [3, 4, 5, 6]] = frames_np[..., [4, 5, 6, 3]]

    # For Tensorboard logging
    writer = SummaryWriter(log_dir=f"{proj_dir}/logdir/{args.run_name}/{args.out_name}")
    writer.add_text("args", str(args.__dict__))
    writer.add_text("remaining_args", str(remaining_args))

    logger = my_logging.get_logger(f"{args.out_name}")
    logger.info(f"Starting")

    # Make visuals here
    part_names = ["head", "left hand", "right hand"]
    feat_names = ["pos x", "pos y", "pos z", "rot x", "rot y", "rot z", "rot w"]
    ii = 1
    for i, part_name in enumerate(part_names):
        for j, feat_name in enumerate(feat_names):
            pass

    canvas = scene.SceneCanvas(
        keys="interactive", bgcolor="white", size=(640, 640), show=False
    )
    left_view = scene.widgets.ViewBox(
        parent=canvas.scene,
        border_color="r",
    )
    left_view.camera = scene.TurntableCamera(
        up="z", fov=1.0, elevation=30.0, distance=10, azimuth=-60
    )
    left_view.camera.rect = 0, 0, 1, 1
    visual_data = VisualDataContainer(left_view)

    @canvas.events.resize.connect
    def resize(event=None):
        left_view.pos = 0, 0
        left_view.size = canvas.size[0] // 1, canvas.size[1] // 1

    resize()

    imgs = []
    for frame in tqdm(frames_np):
        part_data = frame
        xyzs = part_data[..., :3]
        quats = part_data[..., 3:]
        visual_data.body_markers[0].set_data(
            # pos=node_positions[t, [6, 11]],
            pos=xyzs,
            face_color=(1, 0, 0, 0.5),
            edge_color=(0, 0, 0, 0),
            size=0.2,
            edge_width=0,
        )

        # quats[..., [0, 1, 2, 3]] = quats[..., [0, 2, 1, 3]]
        # quats[..., 1] *= -1
        # quats[..., :-1] *= -1

        node_rotmats = Rotation.from_quat(quats).as_matrix()
        # node_rotmats[..., [0, 1, 2]] = node_rotmats[..., [0, 2, 1]]
        # node_rotmats[..., 1] *= -1

        # correction = Rotation.from_euler("XYZ", [0, 0, 0]).as_matrix()
        correction = Rotation.from_euler("XYZ", [np.pi / 2, 0, 0]).as_matrix()

        visual_data.sparse_axes[0].transform.reset()
        visual_data.sparse_axes[1].transform.reset()
        visual_data.sparse_axes[2].transform.reset()

        visual_data.sparse_axes[0].transform.translate(xyzs[0])
        visual_data.sparse_axes[1].transform.translate(xyzs[1])
        visual_data.sparse_axes[2].transform.translate(xyzs[2])

        visual_data.sparse_axes[0].transform.matrix[:-1, :-1] = (
            correction @ node_rotmats[0]
        )
        visual_data.sparse_axes[1].transform.matrix[:-1, :-1] = (
            correction @ node_rotmats[1]
        )
        visual_data.sparse_axes[2].transform.matrix[:-1, :-1] = (
            correction @ node_rotmats[2]
        )

        left_view.camera.azimuth = -135
        left_view.camera.elevation = 10
        left_view.camera.distance = 300
        canvas.update()
        img_left = canvas.render()

        left_view.camera.azimuth = -135
        left_view.camera.elevation = 90
        left_view.camera.distance = 300
        canvas.update()
        img_right = canvas.render()

        img_np = np.concatenate([img_left, img_right], axis=1)

        imgs.append(img_np)
        if args.debug_yes:
            plt.figure()
            plt.imshow(img_np)
            plt.show()
            plt.close()

    w = imageio.get_writer(
        "dump/movie.mp4",
        format="FFMPEG",
        mode="I",
        fps=15,
        codec="h264",
        pixelformat="yuv420p",
    )
    canvas.close()
    del visual_data
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
