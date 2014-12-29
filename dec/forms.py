from numpy import *
import itertools

class Form(ndarray):

    def __new__(cls, input_array, grid=None, degree=None, primal=None):
        obj = asarray(input_array).view(cls)
        obj.grid = grid
        obj.degree = degree
        obj.primal = primal
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.grid = getattr(obj, 'grid', None)
        self.degree = getattr(obj, 'degree', None)
        self.primal = getattr(obj, 'primal', None)

def DEC(grid):
    '''
    
    P, R, D, H = DEC(g)

    this function returns the usual dec operators which then select the appropriate
    specialization of the operator for the given form based on its order and whether
    it is primal or dual.
    '''

    ops_1D = operators(1)

    def typed1(op):
        by_codomain = {}
        for tup in ops_1D[op]:
            _, _, codomain = tup
            by_codomain[codomain] = tup
        def O(*arg):
            codomain = arg[-2:]
            f, _, _ = by_codomain[codomain]
            return Form(getattr(grid, f)(arg[0]), grid, *codomain)
        return O

    def typed2(op):
        by_domain = {}
        for tup in ops_1D[op]:
            _, domain, _ = tup
            by_domain[domain] = tup
        def O(form):
            " Select appropriate operator for form"
            assert grid == form.grid
            f, _, codomain = by_domain[(form.degree, form.primal)]
            return Form(getattr(grid, f)(form), grid, *codomain)
        return O

    def typed3(op):
        by_domain = {}
        for tup in ops_1D[op]:
            _, domain, _ = tup
            by_domain[domain] = tup
        def O(form):
            " Select appropriate operator for form"
            assert grid == form.grid
            f, _, _ = by_domain[(form.degree, form.primal)]
            return getattr(grid, f)(form)
        return O

    return (typed1('P'),
            typed3('R'),
            typed2('D'),
            typed2('H'))
    
def operators(n):
    """
    Return all the operators for dimension n together with their domains and codomains.

    >>> (operators(1) == 
    ... {'D': [('D0', (0, True), (1, True)), 
    ...        ('D0d', (0, False), (1, False))],
    ...  'H': [('H0', (0, True), (1, False)),
    ...        ('H0d', (0, False), (1, True)),
    ...        ('H1', (1, True), (0, False)),
    ...        ('H1d', (1, False), (0, True))],
    ...  'P': [('P0', None, (0, True)),
    ...        ('P0d', None, (0, False)),
    ...        ('P1', None, (1, True)),
    ...        ('P1d', None, (1, False))],
    ...  'R': [('R0', (0, True), None),
    ...        ('R0d', (0, False), None),
    ...        ('R1', (1, True), None),
    ...        ('R1d', (1, False), None)]})
    True
    
    """
    name = lambda n, k, t: '{0}{1}{2}'.format(n, k, 'd' if not t else '')
    # enumerate all the possible discrete forms
    def P(tup): (k, t) = tup; return ( name('P', k, t), None, (k, t) )
    def R(tup): (k, t) = tup; return ( name('R', k, t), (k, t), None )
    def D(tup): (k, t) = tup; return ( name('D', k, t), (k, t), (k+1, t) )
    def H(tup): (k, t) = tup; return ( name('H', k, t), (k, t), (n-k, not t) )
    # Add more operators here - Wedge, Contraction/Flat ?
    forms = tuple(itertools.product(range(n+1), (True, False)))
    return dict(P=[P(f) for f in forms],
                R=[R(f) for f in forms],
                D=[D(f) for f in forms if f[0]<n],
                H=[H(f) for f in forms])


