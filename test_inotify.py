import os
from collections import deque

from inotify_simple import INotify, flags


def main():
    inotify = INotify()
    watch_flags = flags.CLOSE_WRITE
    root_dir = 'output'

    # recurse into all subdirectories, adding each into add_watch
    dir_queue = deque()
    dir_queue.append(root_dir)
    while dir_queue:
        current_dir = dir_queue.popleft()
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path):
                dir_queue.append(item_path)
                inotify.add_watch(item_path, watch_flags)

    while True:
        # And see the corresponding events:
        for event in inotify.read():
            print(event)
            for flag in flags.from_mask(event.mask):
                print('    ' + str(flag))


if __name__ == '__main__':
    main()