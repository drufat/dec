import numpy as np


def decform_factory(name):
    
    F = type(name, (object,), {})
    
    def __init__(self, degree, isprimal, grid, array):
        self.array = np.asarray(array)
        self.degree = degree
        self.isprimal = isprimal
        self.grid = grid
    
    def __repr__(self):
        t = (self.degree, self.isprimal, self.grid, self.array)
        return name + t.__repr__()
    
    def __eq__(self, other):
        return (self.degree is other.degree and 
                self.isprimal is other.isprimal and
                self.grid is other.grid and 
                np.allclose(self.array, other.array))
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __xor__(self, other):
        return self.W(other)

    def binary(name):
        def __fname__(self, other):
            if type(other) is type(self):
                assert self.degree is other.degree
                assert self.grid is other.grid
                assert self.isprimal is other.isprimal
                arr = getattr(self.array, name)(other.array)
            else:    
                arr = getattr(self.array, name)(other)
            return F(self.degree, self.isprimal, self.grid, arr)
        return __fname__

    def unary(name):
        def __fname__(self):
            arr = getattr(self.array, name)()
            return F(self.degree, self.isprimal, self.grid, arr)
        return __fname__    

    #################
    @property
    def R(self):
        '''
        Reconstruction
        Discrete Form -> Python Function
        '''
        d, p, g, a = self.degree, self.isprimal, self.grid, self.array
        def func(*x):
            return sum(a[i]*g.dec.B[d, p](i)(*x) for i in range(g.N[d, p]))
        return func

    @property
    def I(self):
        '''
        Interpolation
        Discrete Form -> Python Function
        '''
        d, p, g, a = self.degree, self.isprimal, self.grid, self.array
        return g.dec.I[d, p](a)

    @property
    def Rf(self):
        '''
        Refine
        Discrete Form -> Component Form
        '''
        d, p, g, a = self.degree, self.isprimal, self.grid, self.array
        return g.refine.T[d, p](a)
    ####################
        
    @property
    def D(self):
        '''
        Derivative
        '''
        d, p, g, a = self.degree, self.isprimal, self.grid, self.array
        a = g.dec.D[d, p](a)
        if a is 0: return 0
        return F(d+1, p, g, a)
    
    @property
    def H(self):
        '''
        Hodge Star
        '''
        d, p, g, a = self.degree, self.isprimal, self.grid, self.array
        n = g.dimension
        a = g.dec.H[d, p](a)
        return F(n-d, not p, g, a)
    
    def W(self, other, toprimal=True):
        '''
        Wedge Product
        '''
        d1, p1, g1, a1 = self.degree, self.isprimal, self.grid, self.array
        d2, p2, g2, a2 = other.degree, other.isprimal, other.grid, other.array
        assert g1 is g2
        a = g1.dec.W[(d1, p1), (d2, p2), toprimal](a1, a2)
        return F(d1+d2, toprimal, g1, a)

    def C(self, other, toprimal=True):
        '''
        Contraction
        '''
        d1, p1, g1, a1 = self.degree, self.isprimal, self.grid, self.array
        d2, p2, g2, a2 = other.degree, other.isprimal, other.grid, other.array
        assert g1 is g2 and d1 == 1
        a = g1.dec.C[p1, (d2, p2), toprimal](a1, a2)
        if a is 0: return 0
        return F(d2-1, toprimal, g1, a)

    for m in '''
        __init__
        __repr__
        __eq__
        __ne__
        __repr__
        __xor__
        R I Rf D H W C
        '''.split():
        setattr(F, m, locals()[m])
    for m in '''
            __add__
            __radd__
            __mul__
            __rmul__
            __sub__
            __rsub__
            __div__
            __truediv__
            '''.split():
        setattr(F, m, binary(m))
    for m in '''
            __neg__
            '''.split():
        setattr(F, m, unary(m))
        
    return F

decform = decform_factory('decform')
