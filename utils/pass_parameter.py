import json
import os

file = 'temp'

def save_parameter(_config: dict):
    global file
    with open(file, 'w') as f:
        json.dump(_config, f)

def read_parameter(_key, _remove: bool = False):
    global file
    _config = {}
    if os.path.exists(file):
        with open('temp', 'r') as f:
            _config = json.load(f)
        if _remove:
            os.remove(file)
        return _config.get(_key, None)
    else:
        return None
