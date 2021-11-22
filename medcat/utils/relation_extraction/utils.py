import os
import pickle

def load_bin_file(file_name, path="./"):
    with open(os.path.join(path, file_name), 'rb') as f:
        data = pickle.load(f)
    return data

def save_bin_file(file_name, data, path="./"):
    with open(os.path.join(path, file_name), "wb") as f:
        pickle.dump(data, f)