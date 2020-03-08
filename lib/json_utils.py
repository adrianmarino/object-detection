import json
import numpy as np


def to_json(obj):
    return json.dumps(
        obj,
        default=lambda o: o.tolist() if isinstance(o, (np.ndarray, np.generic)) else o.__dict__,
        sort_keys=True,
        indent=4
    )
