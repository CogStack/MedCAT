from gensim.matutils import unitvec as g_unitvec
import numpy as np

def unitvec(arr):
    return g_unitvec(np.array(arr))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
