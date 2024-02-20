#!/usr/bin/env python3
'''
o-O-o o-O-o o--o    O
  |     |   |   |  / \
  |     |   O--o  o---o
  |     |   |   | |   |
  o   o-O-o o--o  o   o

Painting robot utils.
© Daniel Berio (@colormotor) 2023 - ...

utils - general Python utilitis (performance, files, strings)
'''


import pickle
import os, sys, time, shutil
import argparse

import json
import numpy as np


class perf_timer:
    def __init__(self, name='', verbose=True):
        #if name and verbose:
        #    print(name)
        self.name = name
        self.verbose = verbose
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.t)*1000
        if self.name and self.verbose:
            print('%s: elapsed time %.3f milliseconds'%(self.name, self.elapsed))


### Serialization
def load_pkl(data_path):
    with open(os.path.expanduser(data_path), "rb") as f:
        try:
            raw_data = pickle.load(f)
        except:
            try:
                print('failed normal load, trying latin1')
                raw_data = pickle.load(f, encoding='latin1')
            except:
                print('also failed latin1, trying bytes')
                raw_data = pickle.load(f, encoding='bytes')
    return raw_data


def save_pkl(X, data_path, protocol=4):
    with open(os.path.expanduser(data_path), "wb") as f:
        pickle.dump(X, f, protocol=protocol)


def load_json(path):
    import codecs
    try:
        with codecs.open(path, encoding='utf8') as fp:
            data = json.load(fp)
        return data
    except IOError as err:
        print(err)
        print ("Unable to load json file:" + path)
        return {}


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_json(data, path, encoder=NumpyEncoder):
    with open(path, 'w') as fp:
        #default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        #The above breaks
        json.dump(data, fp, indent=4, sort_keys=True, cls=encoder) #, default=default)


def json_string(data, encoder=NumpyEncoder):
    return json.dumps(data, cls=encoder)


def create_dir(dir):
    """create a directory if it doesn't exist. silently fail if it already does exist"""
    try:
        os.makedirs(dir)
    except OSError:
        if not os.path.isdir(dir):
            raise OSError
    return dir


def files_in_dir(path, exts=[], nohidden=True):
    files = []
    path = os.path.expanduser(path)
    for file in os.listdir(path):
        if (nohidden and
            os.path.basename(file).startswith('.')):
            continue
        if exts:
            if type(exts) != list:
                exts = [exts]
            for ext in exts:
                if file.endswith(ext):
                    files.append(os.path.join(path, file))
                    break
        else:
            files.append(os.path.join(path, file))

    return files
