from argparse import ArgumentParser

import imageio
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from tensorboardX import SummaryWriter
from tqdm import tqdm
from vispy import scene

from mlexp_utils import my_logging
from mlexp_utils.dirs import proj_dir
import numpy as np
import matplotlib.pyplot as plt

from viz.visual_data import BeatSaberVisualDataContainer
from xror.xror import XROR
import vispy

vispy.use("egl")


def main(args, remaining_args):
    if args.debug_yes:
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=12346, stdoutToServer=True, stderrToServer=True, suspend=False)

    # filepath = f"{proj_dir}/dataset/025bcf95-9be6-4620-8d68-6b5903e62aee/163bd29f-3209-4890-936f-3fc77dccd19b.xror"
    # filepath = f"{proj_dir}/dataset/025bcf95-9be6-4620-8d68-6b5903e62aee/284e1d66-7bba-4160-969b-0e3ccd0dc5e6.xror"
    filepath = f"{proj_dir}/data/beatsaber/025bcf95-9be6-4620-8d68-6b5903e62aee/1a0b3892-3b20-4122-918d-2a4fd8c0ed3e.xror"
    with open(filepath, 'rb') as f:
        file = f.read()
    xror = XROR.unpack(file)

    frames_np = np.array(xror.data["frames"])

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

    canvas = scene.SceneCanvas(keys='interactive', bgcolor='white', size=(640, 640), show=False)
    left_view = scene.widgets.ViewBox(parent=canvas.scene, border_color='r', )
    left_view.camera = scene.TurntableCamera(up='z', fov=20.0, elevation=30., distance=10, azimuth=-60)
    left_view.camera.rect = 0, 0, 1, 1
    visual_data = BeatSaberVisualDataContainer(left_view)

    @canvas.events.resize.connect
    def resize(event=None):
        left_view.pos = 0, 0
        left_view.size = canvas.size[0] // 1, canvas.size[1] // 1

    resize()

    # Tracking head
    head_sphere = visual_data.body_spheres[0]
    left_hand_sphere = visual_data.body_spheres[1]
    right_hand_sphere = visual_data.body_spheres[2]
    imgs = []
    for frame in tqdm(frames_np[1::4]):
        timestamp = frame[0]
        part_data = frame[1:].reshape(3, -1)
        part_data[..., [0, 1, 2]] = part_data[..., [2, 0, 1]]
        part_data[..., 0] *= -1
        for body_i in range(3):
            visual_data.body_spheres[body_i].transform.translate = part_data[body_i, 0], part_data[body_i, 1], part_data[body_i, 2]
            visual_data.shadow_ellipses[body_i].center = part_data[body_i, 0], part_data[body_i, 1]

        unit_vec = np.array([1, 0, 0]) * 0.5

        my_quat = part_data[:, 3:] * 1  # quaternion from Unity order
        my_quat = my_quat[..., [2, 0, 1, 3]]

        left_right_hand_rotations = Rotation.from_quat(my_quat).as_matrix()
        left_right_hand_rotations[:, 0] *= -1
        unit_vec_rotated = left_right_hand_rotations @ unit_vec

        saber_pos = np.concatenate([part_data[1:, :3], part_data[1:, :3] + unit_vec_rotated[1:, :3]], axis=0)
        visual_data.saber_lines.set_data(
            pos=saber_pos,
            color=(1, 0, 0, 1),
            width=1,
            connect=np.array([[0, 2], [1, 3]])
        )

        shadow_pos = saber_pos * 1
        shadow_pos[..., 2] = 0
        visual_data.shadow_lines.set_data(
            pos=shadow_pos,
            color=(0, 0, 0, 0.5),
            width=1,
            connect=np.array([[0, 2], [1, 3]])
        )


        canvas.update()
        img_np = canvas.render()
        imgs.append(img_np)
        plt.figure()
        plt.imshow(img_np)
        plt.show()

    # writer.add_image(f"HeadAndHandsShouldLookLikeBeatSaber", img_np, 0, dataformats="HWC")
    # vid = np.stack(imgs)[None]
    # writer.add_video(f"HeadAndHandsShouldLookLikeBeatSaber", vid, 0, dataformats="NTHWC", fps=15)

    w = imageio.get_writer(
        "dump/movie.mp4",
        format='FFMPEG',
        mode='I',
        fps=15,
        codec='h264',
        pixelformat='yuv420p'
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
    parser.add_argument("--debug_yes", "-d", action="store_true")  # if set, will pause the program
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
