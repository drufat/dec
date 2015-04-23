'''
Discrete Exterior Calculus Module
==================================
'''
import os, json, numpy
from multipledispatch import dispatch
from functools import partial
dispatch_namespace = dict()

d_ = partial(dispatch, namespace=dispatch_namespace)

def get_data_dir():
    root = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(root, 'data')

def get_data(name):
    '''
    Load data samples provided by file `name` in dec/data
    '''
    filename = os.path.join(get_data_dir(), name)
    with open(filename, 'r') as f:
        return json.load(f)

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def store_data(name, obj):
    '''
    Store `obj` in a file `name` in dec/data.
    Obj must be json serializable.
    '''
    filename = os.path.join(get_data_dir(), name)
    with open(filename, 'w') as f:
        json.dump(obj, f, cls=NumpyAwareJSONEncoder)

