import dec.symbolic 
import itertools
from dec.helper import bunch, slow_integration

Π = lambda *x: tuple(itertools.product(*x))

def wrap_refine(to_refine, from_refine):
    
    T0, T1, T0d, T1d = to_refine()
    U0, U1, U0d, U1d = from_refine()    
    T = {(0, True):  T0, 
         (1, True):  T1, 
         (0, False): T0d, 
         (1, False): T1d,}
    U = {(0, True):  U0, 
         (1, True):  U1, 
         (0, False): U0d, 
         (1, False): U1d,}

    return bunch(T=T, U=U)

def projection(simp):

    P={(0, True)  : lambda f: f(simp[0, True]), 
       (0, False) : lambda f: f(simp[0, False]),

       (1, True)  : lambda f: slow_integration(simp[1, True][0],
                                               simp[1, True][1], f),
       (1, False) : lambda f: slow_integration(simp[1, False][0],
                                               simp[1, False][1], f),} 
    return P

def wedge(refine):

    T, U = refine.T, refine.U
    p = Π((0, 1), (True, False))
    
    Ws = dec.symbolic.wedge_1d()
    W = {}    
    
    def get_w(d0, p0, d1, p1, p2):
        def w(a, b):
            a = T[d0, p0](a)
            b = T[d1, p1](b)
            (c,) = Ws[d0, d1]((a,), (b,))
            return U[d0+d1, p2](c)
        return w

    for ((d0, p0), (d1, p1), p2) in Π(p, p,(True, False)):
        if d0 + d1 > 1: continue
        if p0==p1==p2 and d0==d1==0:
            #no refinement necessary, just multiply directly
            W[(d0, p0), (d1, p1), p2] = lambda a, b: a*b
            continue
        W[(d0, p0), (d1, p1), p2] = get_w(d0, p0, d1, p1, p2)    

    return W

def contraction(refine):

    T, U = refine.T, refine.U        
    p = Π((0, 1), (True, False))
    
    Cs = dec.symbolic.contraction_1d()        
    C = {}
    
    def get_c(p0, d1, p1, p2):
        def c(a, b):
            a = T[1, p0](a)
            b = T[d1, p1](b)
            (c,) = Cs[d1]((a,), (b,))
            return U[d1-1, p2](c)
        return c

    for (p0, (d1, p1), p2) in Π((True, False), p, (True, False)):
        if d1-1 < 0: continue
        C[p0, (d1, p1), p2] = get_c(p0, d1, p1, p2)    

    return C
  
