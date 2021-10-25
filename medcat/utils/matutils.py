from gensim.matutils import unitvec as g_unitvec
import numpy as np


def unitvec(arr):
    return g_unitvec(np.array(arr))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def intersect_nonempty_set(base, target):
    if isinstance(target, set) and target:
        result = base.intersection(target)
    else:
        result = base

    # If result is empty set to empty
    if not result:
        result = {'empty'}

    return result
