import os, json, numpy

def get_data_dir():
    root = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(root, 'data')

class JSONEncoder(json.JSONEncoder):
    
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def getitem(name):
    '''
    Load data samples provided by file `name` in dec/data
    '''
    filename = os.path.join(get_data_dir(), name)
    with open(filename, 'r') as f:
        return json.load(f)

def setitem(name, value):
    '''
    Store `value` in a file `name` in dec/data.
    Must be json serializable.
    '''
    filename = os.path.join(get_data_dir(), name)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'w') as f:
        json.dump(value, f, 
                  cls=JSONEncoder,
                  sort_keys=True,
                  indent=2,)

def delitem(name):
    filename = os.path.join(get_data_dir(), name)
    if os.path.isdir(filename):
        os.rmdir(filename)
    else:
        os.remove(filename)

class Data(object):
    '''
    >>> d = Data()

    >>> data = dict(a='b', c='d', e=[1,2,3,4])
    >>> d['test/test.json'] = data
    >>> assert d['test/test.json'] == data

    >>> a = numpy.arange(100)
    >>> d['test/array.json'] = a
    >>> assert (d['test/array.json'] == a).all()
   
    >>> import sympy
    >>> expr = (sympy.sympify('Integral(exp(x), x)'),
    ...         sympy.sympify('Integral(exp(x+y), (x,0,1), (y,0,1))'))
    >>> d['test/sympy.json'] = repr(expr)
    >>> assert d['test/sympy.json'] == repr(expr)
    
    >>> del d['test/test.json']
    >>> del d['test/array.json']
    >>> del d['test/sympy.json']
    >>> del d['test']
    '''
    
    def __getitem__(self, name):
        return getitem(name)

    def __setitem__(self, name, value):
        return setitem(name, value)

    def __delitem__(self, name):
        delitem(name)
