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

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion
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

    filepath = f"{proj_dir}/data/ours/{args.in_name}.pkl"
    my_posrot = torch.load(filepath, map_location="cpu")
    deproj = torch.load("deproj_rotmats.pkl")

    my_pos = my_posrot[:, :, :3]
    my_quats = my_posrot[:, :, 3:]
    my_rotmats = Rotation.from_quat(my_quats.reshape(-1, 4)).as_matrix()
    my_rotmats = my_rotmats.reshape((my_quats.shape[0], 3, 3, 3))
    their_rotmats = deproj @ my_rotmats
    their_quats = Rotation.from_matrix(their_rotmats.reshape(-1, 3, 3)).as_quat()
    their_quats = their_quats.reshape((my_quats.shape[0], 3, 4))
    their_quats[..., [0, 1, 2, 3]] = their_quats[..., [3, 0, 1, 2]]

    their_posrot = np.concatenate([my_pos, their_quats], axis=-1)
    imgs = []
    for t in range(their_posrot.shape[0]):
        visual_data.body_markers[0].set_data(
            pos=their_posrot[t, :, :3],
            face_color=(1, 0, 0, 1),
            edge_color=(0, 0, 0, 0),
            size=0.2,
            edge_width=0,
        )
        visual_data.sparse_axes[0].visible = True
        visual_data.sparse_axes[1].visible = True
        visual_data.sparse_axes[2].visible = True

        visual_data.sparse_axes[0].transform.reset()
        visual_data.sparse_axes[1].transform.reset()
        visual_data.sparse_axes[2].transform.reset()

        visual_data.sparse_axes[0].transform.translate(their_posrot[t, 0, :3])
        visual_data.sparse_axes[1].transform.translate(their_posrot[t, 1, :3])
        visual_data.sparse_axes[2].transform.translate(their_posrot[t, 2, :3])

        their_matrot = Rotation.from_quat(their_posrot[t, :, 3:]).as_matrix()

        visual_data.sparse_axes[0].transform.matrix[:-1, :-1] = their_matrot[0]
        visual_data.sparse_axes[1].transform.matrix[:-1, :-1] = their_matrot[1]
        visual_data.sparse_axes[2].transform.matrix[:-1, :-1] = their_matrot[2]

        # left_view.camera.azimuth = -135
        # left_view.camera.elevation = 10
        # left_view.camera.distance = 300
        # canvas.update()
        # img = canvas.render()
        # imgs.append(img)
        # if args.debug_yes:
        #     plt.figure()
        #     plt.imshow(img)
        #     # plt.imshow(np.concatenate(imgs, axis=0))
        #     plt.tight_layout()
        #     plt.show()
        #     plt.close()

    their_posrot = torch.as_tensor(their_posrot)
    their_posrot[..., 0, 2] -= 0.2
    global_velocity = SkeletonMotion._compute_velocity(
        p=their_posrot[..., :3], time_delta=1 / 30
    )
    global_angular_velocity = SkeletonMotion._compute_angular_velocity(
        r=their_posrot[..., 3:], time_delta=1 / 30
    )
    d = {
        "ref_rb_pos_subset": their_posrot[..., :3],
        "ref_rb_rot_subset": their_posrot[..., 3:],
        "ref_body_vel_subset": global_velocity,
        "ref_body_ang_vel_subset": global_angular_velocity,
    }

    # w = imageio.get_writer(
    #     "dump/converted.mp4",
    #     format="FFMPEG",
    #     mode="I",
    #     fps=15,
    #     codec="h264",
    #     pixelformat="yuv420p",
    # )
    # canvas.close()
    # del visual_data
    # for img in imgs:
    #     w.append_data(img)
    # w.close()

    # plt.close()

    torch.save(d, f"{proj_dir}/data/ours/{args.out_name}.pkl")

    # For Tensorboard logging
    writer = SummaryWriter(log_dir=f"{proj_dir}/logdir/{args.run_name}/{args.out_name}")
    writer.add_text("args", str(args.__dict__))
    writer.add_text("remaining_args", str(remaining_args))

    logger = my_logging.get_logger(f"{args.out_name}")
    logger.info(f"Starting")
    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--in_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument(
        "--debug_yes", "-d", action="store_true"
    )  # if set, will pause the program
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
