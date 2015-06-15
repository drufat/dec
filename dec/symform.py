import sympy as sy
from dec.helper import nCr

def symform_factory(name):
    '''
    
    >>> x, y = sy.symbols('x y')
    >>> f, g, u, v = sy.symbols('f g u v')
    >>> from dec.symbolic import Chart
    >>> c = Chart(x, y)
    
    >>> α = form(1, c, (f, g))
    >>> φ = form(1, c, (u, v))
    
    >>> -φ
    form(1, Chart(x, y), (-u, -v))
    >>> φ + φ
    form(1, Chart(x, y), (2*u, 2*v))
    >>> φ + α
    form(1, Chart(x, y), (f + u, g + v))
    >>> φ - α
    form(1, Chart(x, y), (-f + u, -g + v))
     
    We can use ^ as the wedge product operator.    
    
    >>> assert φ ^ φ == form(2, c, (0,))
    >>> assert α ^ φ == - φ ^ α
    '''
    
    F = type(name, (object,), {})
    
    def __init__(self, degree, grid, components):
        # make sure the form has the correct number of components
        assert degree <= grid.dimension
        assert len(components) == nCr(grid.dimension, degree)
        self.components = tuple(sy.sympify(_) for _ in components)
        self.grid = grid
        self.degree = degree

    def __repr__(self):
        t = (self.degree, self.grid, self.components)
        return name + t.__repr__()

    def __eq__(self, other):
        return (self.degree == other.degree and 
                self.grid == other.grid and
                self.components == other.components)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __rmul__(self, other):
        if callable(other):
            return other(self)
        else:
            return type(self)(self.degree, (c.__rmul__(other) for c in self.components))

    def __xor__(self, other):
        return self.W(other)
    
    def __getitem__(self, k):
        return self.components[k]
        
    def binary(name):
        def __fname__(self, other):
            if type(other) is type(self):
                assert self.degree == other.degree
                assert self.grid == other.grid
                comps = tuple(getattr(s, name)(o) for s, o in zip(self.components, other.components))
            else:    
                comps = tuple(getattr(s, name)(other) for s in self)
            return F(self.degree, self.grid, comps)
        return __fname__

    def unary(name):
        def __fname__(self):
            comps = tuple(getattr(s, name)() for s in self.components)
            return F(self.degree, self.grid, comps)
        return __fname__
    
    @property
    def D(self):
        d, ch, c = self.degree, self.grid, self.components
        c = ch.dec.D[d](c)
        if c is 0: return 0
        return F(d+1, ch, c)
    
    @property
    def H(self):
        d, ch, c = self.degree, self.grid, self.components
        c = ch.dec.H[d](c)
        if c is 0: return 0
        dim = ch.dimension
        return F(dim-d, ch, c)

    def W(self, other):
        d1, ch1, c1 = self.degree, self.grid, self.components
        d2, ch2, c2 = other.degree, other.grid, other.components
        assert ch1 == ch2
        return F(d1+d2, ch1, ch1.dec.W[d1, d2](c1, c2))

    def C(self, other):
        d1, ch1, c1 = self.degree, self.grid, self.components
        assert d1 == 1
        d2, ch2, c2 = other.degree, other.grid, other.components
        assert ch1 == ch2
        c = self.grid.dec.C[d2](c1, c2)
        if c == 0: return 0
        return F(d2-1, ch1, c)
    
    @property
    def lambdify_(self):
        if len(self.components) == 1:
            return sy.lambdify(self.grid.coords, self.components[0], 'numpy')
        else:
            return sy.lambdify(self.grid.coords, self.components, 'numpy')

    #############
    def P(self, g, isprimal):
        return to_discrete(self, g, isprimal)
    #############

    for m in '''
            __init__
            __eq__
            __ne__
            __repr__
            __rmul__
            __xor__
            __getitem__
            P D H W C
            '''.split():
        setattr(F, m, locals()[m])
    for m in '''
            __add__
            __radd__
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
    setattr(F, 'lambdify', lambdify_)

    return F

form = symform_factory('form')

def to_discrete(f, g, isprimal):
    from dec.decform import decform
    
    d, ch, c = f.degree, f.grid, f.components
    assert g.dimension == ch.dimension

    #Symbolic Integration
    integrate = ch.dec.P[d](c)
    λ = sy.lambdify(ch.cell_coords(d), integrate, 'numpy')

    cells = g.cells[d, isprimal]
    #TODO: It may be necessary to compute limits when x0==x1, y0==y1
    if ch.dimension == 1:
        if   d == 0:
            x0 = cells
            a = λ(x0)
        elif d == 1:
            x0, x1 = cells
            a = λ(x0, x1)
    elif ch.dimension == 2:
        if   d == 0:
            x0, y0 = cells
            a = λ(x0, y0)
        elif d == 1:
            (x0, y0), (x1, y1) = cells
            a = λ(x0, y0, x1, y1)
        elif d == 2:
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = cells
            a = (λ(x0, y0, x1, y1, x2, y2) + 
                 λ(x0, y0, x2, y2, x3, y3))
        
    return decform(d, isprimal, g, a)

