import os

def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)

def ensure_dirs(paths):
    """
    create paths by first checking their existence
    :param paths: list of path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def listlistdirs(dir_path):
    IS_DIR = not os.path.isfile(os.path.join(dir_path, os.listdir(dir_path)[0]))
    if IS_DIR:
        fnames = []
        for dir in os.listdir(dir_path):
            for f in os.listdir(os.path.join(dir_path, dir)):
                fname = os.path.join(dir, f)
                fnames.append(fname)
    else:
        fnames = os.listdir(dir_path)
    return fnames