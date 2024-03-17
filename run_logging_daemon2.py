import glob
import os
import time
from argparse import ArgumentParser
from collections import deque

from inotify_simple import INotify, flags


def main(args):

    watch_target_dir = os.path.abspath(args.watch_target_dir)

    already_watched = set()

    while True:
        pkls = set(glob.glob(f"{watch_target_dir}/*.pkl", recursive=True))
        new_pkls = pkls - already_watched
        print(new_pkls)
        for new_pkls in new_pkls:
            os.system(f"python look6.py --out_name asdf --in_posrot_path {new_pkls}")

        # in_posrot_path = os.path.join(wd_path_dict[event.wd], event.name)
        # # os.system(f"python look6.py --out_name asdf --in_posrot_path {in_posrot_path} --debug_yes")
        # os.system(f"python look6.py --out_name asdf --in_posrot_path {in_posrot_path}")
        already_watched = pkls
        time.sleep(300)


if __name__ == '__main__':
    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace(
    #     "localhost", port=12346, stdoutToServer=True, stderrToServer=True, suspend=False
    # )

    parser = ArgumentParser()
    parser.add_argument("--watch_target_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
