import os
from argparse import ArgumentParser
from collections import deque

from inotify_simple import INotify, flags


def main():
    inotify = INotify()
    watch_flags = flags.CLOSE_WRITE
    root_dir = 'output/HumanoidIm/phc_prim_vr'

    # recurse into all subdirectories, adding each into add_watch
    dir_queue = deque()
    dir_queue.append(root_dir)
    wd_path_dict = {}
    while dir_queue:
        current_dir = dir_queue.popleft()
        wd = inotify.add_watch(current_dir, watch_flags)
        wd_path_dict[wd] = current_dir
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path):
                dir_queue.append(item_path)

    print(wd_path_dict)

    while True:
        # And see the corresponding events:
        for event in inotify.read():
            print(event)
            for flag in flags.from_mask(event.mask):
                print('    ' + str(flag))
            if os.path.splitext(event.name)[1] == '.pkl':
                in_posrot_path = os.path.join(wd_path_dict[event.wd], event.name)
                # os.system(f"python look6.py --out_name asdf --in_posrot_path {in_posrot_path} --debug_yes")
                os.system(f"python look6.py --out_name asdf --in_posrot_path {in_posrot_path}")


if __name__ == '__main__':
    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace(
    #     "localhost", port=12346, stdoutToServer=True, stderrToServer=True, suspend=False
    # )

    main(args)
