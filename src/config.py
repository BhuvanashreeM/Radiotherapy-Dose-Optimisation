DATA_PATH = None
MODEL_SAVE_PATH = None
LOG_DIR = None

if DATA_PATH is None:
    raise Exception('Configure data folder location in config.py')

if MODEL_SAVE_PATH  is None:
    raise Exception('Configure model folder location in config.py')

if LOG_DIR is None:
    raise Exception('Configure log folder location in config.py')