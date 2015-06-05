from dec.symbolic import *

try:

    from pythematica import Pythematica
    mathematica = Pythematica()
        
    def Integrate(*args, Assumptions=None):
        return mathematica.Integrate(
                *args, 
                Assumptions=Assumptions)

except ImportError:
    
    def Integrate(*args, Assumptions=None):
        expr, *bounds = args
        return integrate(expr, *reversed(bounds))

def run_mathematica():
    from sympy import sin, cos
    P0, P1, P2 = projections_2d(x, y)

    P1_d = {}    
    for f in ((1,0),
              (0,1),
              (sin(x),0),
              (sin(y),0),
              (-sin(y), sin(x)),
              (sin(x), sin(y)),
              (sin(x), cos(2*y)),
              ):
        P1_d[repr(f)] = repr(P1(f))
        print()
        print(f)
        print(P1_d[repr(f)])
        
    P2_d = {}    
    for f in (1,
              sin(x),
              sin(y),
              cos(x),
              cos(y),
              sin(x+y),
              sin(x+2*y),
              x, y, x**2, y**2, x*y, 
              x**3, x**2*y, x*y**2, y**3,
              ):
        P2_d[repr((f,))] = repr(P2((f,)))
        print()
        print(f)
        print(P2_d[repr((f,))])
    
    dataname = 'memoize/projections.json'
    import dec
    dec.store_data(dataname, {1:P1_d, 2:P2_d})
    return
