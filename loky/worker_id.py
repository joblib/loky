import os


def get_worker_id():
    wid = os.environ.get('LOKY_WORKER_ID', None)
    if wid is None:
        return -1
    return int(wid)
